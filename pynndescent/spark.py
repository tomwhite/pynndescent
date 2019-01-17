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

def build_candidates(sc, current_graph, n_vertices, n_neighbors, max_candidates,
                     rng_state, rho=0.5):
    # spark version
    s = current_graph.shape
    current_graph_rdd = to_rdd(sc, current_graph, (s[0], 4, s[2]))

    def f(current_graph_part):
        # print("current_graph_part", current_graph_part)

        # each part has its own heaps for old and new candidates
        # (TODO: consider making these sparse?)
        new_candidate_neighbors = make_heap(n_vertices, max_candidates)
        old_candidate_neighbors = make_heap(n_vertices, max_candidates)
        n_vertices_part = current_graph_part.shape[1]
        for i in range(n_vertices_part):
            r = np.random.RandomState()
            r.seed(i)
            print("seed spark " + str(i))
            for j in range(n_neighbors):
                if current_graph_part[0, i, j] < 0:
                    continue
                idx = current_graph_part[0, i, j]
                isn = current_graph_part[2, i, j]
                d = r.random_sample()
                if r.random_sample() < rho:
                    c = 0
                    if isn:
                        if i == 1:
                            print("heap_push(new_candidate_neighbors, i, d, idx, isn) spark", i, d, idx, isn)
                        c += heap_push(new_candidate_neighbors, i, d, idx, isn)
                        c += heap_push(new_candidate_neighbors, idx, d, i, isn)
                    else:
                        heap_push(old_candidate_neighbors, i, d, idx, isn)
                        heap_push(old_candidate_neighbors, idx, d, i, isn)

                    if c > 0 :
                        current_graph_part[2, i, j] = 0

        # print("new_candidate_neighbors part", new_candidate_neighbors)
        # TODO: emit (index, new_candidate_neighbors) pairs, then we can reduceByKey, where merge_heaps is the reduce operation
        # TODO: may want to have (index, new_candidate_neighbors, old_candidate_neighbors)
        return new_candidate_neighbors
    all_new_candidate_neighbors = current_graph_rdd.map(f).collect()
    print("all_new_candidate_neighbors", all_new_candidate_neighbors)

    merged_new_candidate_neighbors = merge_heaps(all_new_candidate_neighbors[0], all_new_candidate_neighbors[1])
    # print("merged_new_candidate_neighbors", merged_new_candidate_neighbors)

    return merged_new_candidate_neighbors

def merge_heaps(heap1, heap2):
    # TODO: check heaps have the same size
    s = heap2.shape
    for row in range(s[1]):
        for ind in range(s[2]): # TODO: reverse to make more efficient
            index = heap2[0, row, ind]
            weight = heap2[1, row, ind]
            flag = heap2[2, row, ind]
            heap_push(heap1, row, weight, index, flag)
    return heap1