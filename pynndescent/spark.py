from functools import partial
import math
import numpy as np

from sklearn.utils import check_random_state

from pynndescent import distances
from pynndescent.heap import *
from pynndescent.utils import *

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

# Chunking functions
# Could use Zarr/Zappy/Anndata for these

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

def to_local_rows(sc, arr, chunks):
    func, chunk_indices = read_chunks(arr, chunks)
    return [func(i) for i in chunk_indices]

# NNDescent algorithm

def get_rng_state(random_state):
    random_state = check_random_state(random_state)
    return random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

def init_current_graph(data, n_neighbors, rng_state):
    # This is just a copy from make_nn_descent -> nn_descent
    r = np.random.RandomState()

    dist = distances.named_distances['euclidean']

    current_graph = make_heap(data.shape[0], n_neighbors)
    # for each row i
    for i in range(data.shape[0]):
        # choose K rows from the whole matrix
        r.seed(i)
        indices = rejection_sample2(n_neighbors, data.shape[0], r)
        # and work out the dist from row i to each of the random K rows
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)

    return current_graph

def init_current_graph_rdd(sc, data, n_neighbors, rng_state):
    dist = distances.named_distances['euclidean']
    n_vertices = data.shape[0]
    chunk_size = 4
    current_graph_rdd = make_heap_rdd(sc, n_vertices, n_neighbors, chunk_size)
    current_graph_chunks = (3, chunk_size, n_neighbors) # 3 is first heap dimension

    def init_current_graph_for_each_part(index, iterator):
        r = np.random.RandomState()
        offset = index * chunk_size
        for current_graph_part in iterator:
            n_vertices_part = current_graph_part.shape[1]
            # Each part has its own heap for the current graph, which
            # are combined in the reduce stage.
            current_graph_local = make_heap(n_vertices, n_neighbors)
            for i in range(n_vertices_part):
                iabs = i + offset
                r.seed(iabs)
                indices = rejection_sample2(n_neighbors, n_vertices, r)
                for j in range(indices.shape[0]):
                    d = dist(data[iabs], data[indices[j]])
                    heap_push(current_graph_local, iabs, d, indices[j], 1)
                    heap_push(current_graph_local, indices[j], d, iabs, 1)

            # Split current_graph into chunks and return each chunk keyed by its index.
            read_chunk_func_new, chunk_indices = read_chunks(current_graph_local, current_graph_chunks)
            for i, chunk_index in enumerate(chunk_indices):
                yield i, read_chunk_func_new(chunk_index)

    return current_graph_rdd \
        .mapPartitionsWithIndex(init_current_graph_for_each_part) \
        .reduceByKey(merge_heaps) \
        .values()

def build_candidates_rdd(sc, current_graph_rdd, n_vertices, n_neighbors, max_candidates,
                     rng_state, rho=0.5):
    chunk_size = 4
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
        .values()

    return candidate_neighbors_combined

def nn_descent(sc, data, n_neighbors, rng_state, max_candidates=50,
               n_iters=10, delta=0.001, rho=0.5,
               rp_tree_init=False, leaf_array=None, verbose=False):

    dist = distances.named_distances['euclidean']

    n_vertices = data.shape[0]
    chunk_size = 4
    current_graph_chunks = (3, chunk_size, n_neighbors) # 3 is first heap dimension

    current_graph_rdd = init_current_graph_rdd(sc, data, n_neighbors, rng_state)

    for n in range(n_iters):

        candidate_neighbors_combined = build_candidates_rdd(sc, current_graph_rdd,
                                                     n_vertices,
                                                     n_neighbors,
                                                     max_candidates,
                                                     rng_state, rho)


        def nn_descent_for_each_part(index, iterator):
            offset = index * chunk_size
            for candidate_neighbors_combined_part in iterator:
                new_candidate_neighbors_part, old_candidate_neighbors_part = candidate_neighbors_combined_part
                n_vertices_part = new_candidate_neighbors_part.shape[1]
                # Each part has its own heaps for the current graph, which
                # are combined in the reduce stage.
                current_graph = make_heap(n_vertices, n_neighbors)
                c = 0 # not used yet (needs combining across all partitions)
                for i in range(n_vertices_part):
                    for j in range(max_candidates):
                        p = int(new_candidate_neighbors_part[0, i, j])
                        if p < 0:
                            continue
                        for k in range(j, max_candidates):
                            q = int(new_candidate_neighbors_part[0, i, k])
                            if q < 0:
                                continue

                            d = dist(data[p], data[q])
                            c += heap_push(current_graph, p, d, q, 1)
                            c += heap_push(current_graph, q, d, p, 1)

                        for k in range(max_candidates):
                            q = int(old_candidate_neighbors_part[0, i, k])
                            if q < 0:
                                continue

                            d = dist(data[p], data[q])
                            c += heap_push(current_graph, p, d, q, 1)
                            c += heap_push(current_graph, q, d, p, 1)

                # Split current_graph into chunks and return each chunk keyed by its index.
                read_chunk_func, chunk_indices = read_chunks(current_graph, current_graph_chunks)
                for i, chunk_index in enumerate(chunk_indices):
                    yield i, read_chunk_func(chunk_index)

        current_graph_rdd_updates = candidate_neighbors_combined\
            .mapPartitionsWithIndex(nn_descent_for_each_part)\
            .reduceByKey(merge_heaps)\
            .values()

        # merge the updates into the current graph
        current_graph_rdd = current_graph_rdd\
            .zip(current_graph_rdd_updates)\
            .map(lambda pair: merge_heaps(pair[0], pair[1]))

        # TODO: transfer c back from each partition and sum, in order to implement termination criterion
        # if c <= delta * n_neighbors * data.shape[0]:
        #     break

    # stack results (again, shouldn't collect result, but instead save to storage)
    current_graph = np.hstack(current_graph_rdd.collect())

    return deheap_sort(current_graph)