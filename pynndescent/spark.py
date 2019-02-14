from functools import partial
import math
import numpy as np

from pynndescent.heap import make_heap_rdd
from pynndescent.threaded import chunk_rows, sort_heap_updates, chunk_heap_updates, current_graph_map_jit, apply_heap_updates_jit, apply_new_and_old_heap_updates_jit, candidates_map_jit, nn_descent_map_jit
from pynndescent.utils import deheap_sort, make_heap

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

def init_current_graph_rdd(sc, data_broadcast, data_shape, n_neighbors, chunk_size):
    n_vertices = data_shape[0]
    current_graph_rdd = make_heap_rdd(sc, n_vertices, n_neighbors, chunk_size)

    def create_heap_updates(index, iterator):
        data_local = data_broadcast.value
        for _ in iterator:
            # Each part has its own heap updates for the current graph, which
            # are combined in the reduce stage.
            max_heap_update_count = chunk_size * n_neighbors * 2
            heap_updates_local = np.zeros((max_heap_update_count, 4))
            rows = chunk_rows(chunk_size, index, n_vertices)
            count = current_graph_map_jit(rows, n_vertices, n_neighbors, data_local, heap_updates_local)
            heap_updates_local = sort_heap_updates(heap_updates_local, count)
            offsets = chunk_heap_updates(heap_updates_local, n_vertices, chunk_size)
            # Split updates into chunks and return each chunk keyed by its index.
            for i in range(len(offsets) - 1):
                yield i, heap_updates_local[offsets[i]:offsets[i+1]]

    def update_heap(index, iterator):
        for i in iterator:
            heap, updates = i
            for u in updates:
                offset = index * chunk_size
                apply_heap_updates_jit(heap, u, offset)
            yield heap

    # do a group by (rather than some aggregation function), since we don't combine/aggregate on the map side
    # use a partition function that ensures that partition index i goes to partition i, so we can zip with the original heap
    updates_rdd = current_graph_rdd \
        .mapPartitionsWithIndex(create_heap_updates) \
        .groupByKey(partitionFunc=lambda i: i) \
        .values()

    # merge the updates into the current graph
    return current_graph_rdd \
        .zip(updates_rdd) \
        .mapPartitionsWithIndex(update_heap)

def build_candidates_rdd(current_graph_rdd, n_vertices, n_neighbors, max_candidates, chunk_size,
                     rng_state, rho=0.5):

    def create_heap_updates(index, iterator):
        for current_graph_part in iterator:
            # Each part has its own heap updates for the current graph, which
            # are combined in the reduce stage.
            max_heap_update_count = chunk_size * n_neighbors * 2
            heap_updates_local = np.zeros((max_heap_update_count, 5))
            rows = chunk_rows(chunk_size, index, n_vertices)
            offset = chunk_size * index
            count = candidates_map_jit(rows, n_neighbors, current_graph_part, heap_updates_local, offset)
            heap_updates_local = sort_heap_updates(heap_updates_local, count)
            offsets = chunk_heap_updates(heap_updates_local, n_vertices, chunk_size)
            # Split updates into chunks and return each chunk keyed by its index.
            for i in range(len(offsets) - 1):
                yield i, heap_updates_local[offsets[i]:offsets[i+1]]

    def update_heaps(index, iterator):
        for i in iterator:
            current_graph_part, updates = i
            part_size = chunk_size
            if n_vertices % chunk_size != 0  and index == n_vertices // chunk_size:
                part_size = n_vertices % chunk_size
            new_candidate_neighbors_part = make_heap(part_size, max_candidates)
            old_candidate_neighbors_part = make_heap(part_size, max_candidates)
            for u in updates:
                offset = index * chunk_size
                apply_new_and_old_heap_updates_jit(current_graph_part, new_candidate_neighbors_part, old_candidate_neighbors_part, u, offset)
            yield new_candidate_neighbors_part, old_candidate_neighbors_part

    updates_rdd = current_graph_rdd \
        .mapPartitionsWithIndex(create_heap_updates) \
        .groupByKey(partitionFunc=lambda i: i) \
        .values()

    return current_graph_rdd \
        .zip(updates_rdd) \
        .mapPartitionsWithIndex(update_heaps)

def nn_descent(sc, data, n_neighbors, rng_state, chunk_size, max_candidates=50,
               n_iters=10, delta=0.001, rho=0.5,
               rp_tree_init=False, leaf_array=None, verbose=False):

    data_broadcast = sc.broadcast(data)

    n_vertices = data.shape[0]

    current_graph_rdd = init_current_graph_rdd(sc, data_broadcast, data.shape, n_neighbors, chunk_size)

    for n in range(n_iters):

        candidate_neighbors_combined = build_candidates_rdd(current_graph_rdd,
                                                     n_vertices,
                                                     n_neighbors,
                                                     max_candidates,
                                                     chunk_size,
                                                     rng_state, rho)


        def create_heap_updates(index, iterator):
            data_local = data_broadcast.value
            for candidate_neighbors_combined_part in iterator:
                new_candidate_neighbors_part, old_candidate_neighbors_part = candidate_neighbors_combined_part
                # Each part has its own heap updates for the current graph, which
                # are combined in the reduce stage.
                max_heap_update_count = chunk_size * max_candidates * max_candidates * 4
                heap_updates_local = np.zeros((max_heap_update_count, 4))
                rows = chunk_rows(chunk_size, index, n_vertices)
                offset = chunk_size * index
                count = nn_descent_map_jit(rows, max_candidates, data_local, new_candidate_neighbors_part, old_candidate_neighbors_part, heap_updates_local, offset)
                heap_updates_local = sort_heap_updates(heap_updates_local, count)
                offsets = chunk_heap_updates(heap_updates_local, n_vertices, chunk_size)
                # Split updates into chunks and return each chunk keyed by its index.
                for i in range(len(offsets) - 1):
                    yield i, heap_updates_local[offsets[i]:offsets[i+1]]

        def update_heap(index, iterator):
            for i in iterator:
                heap, updates = i
                for u in updates:
                    offset = index * chunk_size
                    apply_heap_updates_jit(heap, u, offset)
                yield heap

        updates_rdd = candidate_neighbors_combined \
            .mapPartitionsWithIndex(create_heap_updates) \
            .groupByKey(partitionFunc=lambda i: i) \
            .values()

        current_graph_rdd = current_graph_rdd \
            .zip(updates_rdd) \
            .mapPartitionsWithIndex(update_heap)

        # TODO: transfer c back from each partition and sum, in order to implement termination criterion
        # if c <= delta * n_neighbors * data.shape[0]:
        #     break

    # stack results (again, shouldn't collect result, but instead save to storage)
    current_graph = np.hstack(current_graph_rdd.collect())

    return deheap_sort(current_graph)