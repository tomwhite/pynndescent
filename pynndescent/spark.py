from functools import partial
import math
import numpy as np

from sklearn.utils import check_random_state

from pynndescent import distances
from pynndescent.utils import *

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

def read_zarr_chunk(arr, chunks, chunk_index):
    return arr[
           chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1),
           chunks[1] * chunk_index[1] : chunks[1] * (chunk_index[1] + 1),
           chunks[2] * chunk_index[2] : chunks[2] * (chunk_index[2] + 1),
           ]

def get_chunk_indices(shape, chunks):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    """
    return [
        (i, j, k)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
        for k in range(int(math.ceil(float(shape[2]) / chunks[2])))
    ]

def read_chunks(arr, chunks):
    shape = arr.shape
    func = partial(read_zarr_chunk, arr, chunks)
    chunk_indices = get_chunk_indices(shape, chunks)
    return func, chunk_indices

def to_rdd(sc, arr, chunks):
    func, chunk_indices = read_chunks(arr, chunks)
    local_rows = [func(i) for i in chunk_indices]
    return sc.parallelize(local_rows, len(local_rows))

def to_local_rows(sc, arr, chunks):
    func, chunk_indices = read_chunks(arr, chunks)
    return [func(i) for i in chunk_indices]

def get_rng_state(random_state):
    random_state = check_random_state(random_state)
    return random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

def init_current_graph(data, n_neighbors, random_state):
    # This is just a copy from make_nn_descent -> nn_descent

    # TODO: parallelize this

    dist = distances.named_distances['euclidean']

    rng_state = get_rng_state(random_state)

    current_graph = make_heap(data.shape[0], n_neighbors)
    # for each row i
    for i in range(data.shape[0]):
        # choose K rows from the whole matrix
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        print(indices)
        # and work out the dist from row i to each of the random K rows
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)

    return current_graph

# spark version
def build_candidates(sc, current_graph, n_vertices, n_neighbors, max_candidates,
                     rng_state, rho=0.5):
    s = current_graph.shape
    chunk_size = 4
    current_graph_rdd = to_rdd(sc, current_graph, (s[0], chunk_size, s[2]))

    candidate_chunks = (3, chunk_size, max_candidates) # 3 is first heap dimension

    def build_candidates_for_each_part(index, iterator):
        offset = index * chunk_size
        for current_graph_part in iterator:
            n_vertices_part = current_graph_part.shape[1]
            # Each part has its own heaps for old and new candidates, which
            # are combined in the reduce stage.
            # (TODO: make these sparse - use COO (or maybe LIL) to construct, then convert to CSR to slice)
            new_candidate_neighbors = make_heap(n_vertices, max_candidates)
            old_candidate_neighbors = make_heap(n_vertices, max_candidates)
            for i in range(n_vertices_part):
                iabs = i + offset
                r = np.random.RandomState()
                r.seed(iabs)
                for j in range(n_neighbors):
                    if current_graph_part[0, i, j] < 0:
                        continue
                    idx = current_graph_part[0, i, j]
                    isn = current_graph_part[2, i, j]
                    d = r.random_sample()
                    if r.random_sample() < rho:
                        c = 0
                        if isn:
                            c += heap_push(new_candidate_neighbors, iabs, d, idx, isn)
                            c += heap_push(new_candidate_neighbors, idx, d, iabs, isn)
                        else:
                            heap_push(old_candidate_neighbors, iabs, d, idx, isn)
                            heap_push(old_candidate_neighbors, idx, d, iabs, isn)

                        if c > 0 :
                            current_graph_part[2, i, j] = 0

            # Split candidate_neighbors into chunks and return each chunk keyed by its index.
            # New and old are the same size, so chunk indices are the same.
            read_chunk_func_new, chunk_indices = read_chunks(new_candidate_neighbors, candidate_chunks)
            read_chunk_func_old, chunk_indices = read_chunks(old_candidate_neighbors, candidate_chunks)
            for i, chunk_index in enumerate(chunk_indices):
                yield i, (read_chunk_func_new(chunk_index), read_chunk_func_old(chunk_index))

    candidate_neighbors_combined = current_graph_rdd\
        .mapPartitionsWithIndex(build_candidates_for_each_part)\
        .reduceByKey(merge_heap_pairs)\
        .values()\
        .collect()

    # stack results (this should really be materialized to a store, e.g. as Zarr)
    new_candidate_neighbors_combined = np.hstack([pair[0] for pair in candidate_neighbors_combined])
    old_candidate_neighbors_combined = np.hstack([pair[1] for pair in candidate_neighbors_combined])

    return new_candidate_neighbors_combined, old_candidate_neighbors_combined

def merge_heap_pairs(heap_pair1, heap_pair2):
    heap1_new, heap1_old = heap_pair1
    heap2_new, heap2_old = heap_pair2
    return merge_heaps(heap1_new, heap2_new), merge_heaps(heap1_old, heap2_old)

def merge_heaps(heap1, heap2):
    heap = heap1.copy()
    # TODO: check heaps have the same size
    s = heap2.shape
    for row in range(s[1]):
        for ind in range(s[2]): # TODO: reverse to make more efficient?
            index = heap2[0, row, ind]
            weight = heap2[1, row, ind]
            flag = heap2[2, row, ind]
            heap_push(heap, row, weight, index, flag)
    return heap