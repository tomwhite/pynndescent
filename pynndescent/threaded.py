import concurrent.futures
from functools import partial
import math
import numba
import numpy as np

from pynndescent import distances
#from pynndescent.heap import *
from pynndescent.utils import deheap_sort, heap_push, make_heap, rejection_sample, seed, tau_rand

# NNDescent algorithm

def init_current_graph(data, n_neighbors):
    # This is just a copy from make_nn_descent -> nn_descent
    rng_state = np.empty((3,), dtype=np.int64)
    dist = distances.named_distances['euclidean']

    current_graph = make_heap(data.shape[0], n_neighbors)
    # for each row i
    for i in range(data.shape[0]):
        # choose K rows from the whole matrix
        seed(rng_state, i)
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        # and work out the dist from row i to each of the random K rows
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)

    return current_graph

@numba.njit('i8[:](i8, i8, i8)')
def chunk_rows(chunk_size, index, n_vertices):
    return np.arange(chunk_size * index, min(chunk_size * (index + 1), n_vertices))

@numba.njit('i8[:](f8[:, :], i8)')
def sort_heap_updates(heap_updates, num_heap_updates):
    """Take an array of unsorted heap updates and return sorted indices
    (from argsort)."""
    row_numbers = heap_updates[:num_heap_updates, 0]
    return row_numbers.argsort()

# Numba JIT doesn't work since `searchsorted` with the `sorter` arg is not supported.
# See https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#other-functions
#@numba.njit('i8[:](f8[:, :], i8, i8[:], i8)')
def chunk_heap_updates(heap_updates, num_heap_updates, sorter, chunk_size):
    """Take an array of unsorted heap updates and return offsets for each chunk."""
    row_numbers = heap_updates[:num_heap_updates, 0]
    chunk_boundaries = [i * chunk_size for i in range(int(math.ceil(float(num_heap_updates) / chunk_size)) + 1)]
    return np.searchsorted(row_numbers, chunk_boundaries, side='left', sorter=sorter)

@numba.njit('i8[:](f8[:, :], i8, i8, i8)')
def sort_and_chunk_heap_updates(heap_updates, num_heap_updates, chunk_size, n_vertices):
    """Take an array of unsorted heap updates and sort in-place (by row number)
    and return the offsets for each chunk."""
    row_numbers = heap_updates[:num_heap_updates, 0]
    heap_updates = heap_updates[:num_heap_updates][row_numbers.argsort()]
    chunk_boundaries = [i * chunk_size for i in range(int(math.ceil(float(n_vertices) / chunk_size)) + 1)]
    offsets = np.searchsorted(heap_updates[:, 0], chunk_boundaries, side='left')
    return offsets

@numba.njit('void(f8[:, :, :], i8[:], i8[:, :], i8, i8, i8)', nogil=True)
def shuffle_jit(heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index):
    o = sort_and_chunk_heap_updates(heap_updates[index], heap_update_counts[index], chunk_size, n_vertices)
    offsets[index, :o.shape[0]] = o

# Map Reduce functions to be jitted

dist = distances.named_distances['euclidean']

@numba.njit('i8(i8[:], i8, i8, f4[:, :], f8[:, :])', nogil=True)
def current_graph_map_jit(rows, n_vertices, n_neighbors, data, heap_updates):
    rng_state = np.empty((3,), dtype=np.int64)
    count = 0
    for i in rows:
        seed(rng_state, i)
        indices = rejection_sample(n_neighbors, n_vertices, rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_updates[count] = np.array([i, d, indices[j], 1])
            count += 1
            heap_updates[count] = np.array([indices[j], d, i, 1])
            count += 1
    return count

@numba.njit('void(i8, f8[:, :, :], f8[:, :, :], i8[:, :], i8)', nogil=True)
def current_graph_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index):
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            heap_push(current_graph, int(heap_update[0]), heap_update[1], int(heap_update[2]), int(heap_update[3]))

def init_current_graph_threaded(data, n_neighbors, chunk_size=4, threads=2):

    n_vertices = data.shape[0]
    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    current_graph = make_heap(n_vertices, n_neighbors)

    # store the updates in an array
    max_heap_update_count = chunk_size * n_neighbors * 2
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 4))
    heap_update_counts = np.zeros((n_tasks,), dtype=int)

    def current_graph_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return index, current_graph_map_jit(rows, n_vertices, n_neighbors, data, heap_updates[index])

    def current_graph_reduce(index):
        return current_graph_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
    # run map functions
    for index, count in executor.map(current_graph_map, range(n_tasks)):
        heap_update_counts[index] = count

    # sort and chunk heap updates so they can be applied in the reduce
    max_count = heap_update_counts.max()
    offsets = np.zeros((n_tasks, max_count), dtype=int)

    def shuffle(index):
        return shuffle_jit(heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index)

    for _ in executor.map(shuffle, range(n_tasks)):
        pass

    # then run reduce functions
    for _ in executor.map(current_graph_reduce, range(n_tasks)):
        pass

    return current_graph

rho = 0.5 # TODO: pass

@numba.njit('i8(i8[:], i8, f8[:, :, :], f8[:, :])', nogil=True)
def candidates_map_jit(rows, n_neighbors, current_graph, heap_updates):
    rng_state = np.empty((3,), dtype=np.int64)
    count = 0
    for i in rows:
        seed(rng_state, i)
        for j in range(n_neighbors):
            if current_graph[0, i, j] < 0:
                continue
            idx = current_graph[0, i, j]
            isn = current_graph[2, i, j]
            d = tau_rand(rng_state)
            if tau_rand(rng_state) < rho:
                # updates are common to old and new - decided by 'isn' flag
                heap_updates[count] = np.array([i, d, idx, isn, j])
                count += 1
                heap_updates[count] = np.array([idx, d, i, isn, j])
                count += 1
    return count

@numba.njit('void(i8, f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], i8[:, :], i8)', nogil=True)
def candidates_reduce_jit(n_tasks, current_graph, new_candidate_neighbors, old_candidate_neighbors, heap_updates, offsets, index):
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            if heap_update[3]:
                c = heap_push(new_candidate_neighbors, int(heap_update[0]), heap_update[1], int(heap_update[2]), int(heap_update[3]))
                if c > 0:
                    current_graph[2, int(heap_update[0]), int(heap_update[4])] = 0
            else:
                heap_push(old_candidate_neighbors, int(heap_update[0]), heap_update[1], int(heap_update[2]), int(heap_update[3]))

def build_candidates_threaded(current_graph, n_vertices, n_neighbors, max_candidates,
                         rng_state, rho=0.5, chunk_size=4, threads=2):

    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    new_candidate_neighbors = make_heap(n_vertices, max_candidates)
    old_candidate_neighbors = make_heap(n_vertices, max_candidates)

    # store the updates in an array
    max_heap_update_count = chunk_size * n_neighbors * 2
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 5))
    heap_update_counts = np.zeros((n_tasks,), dtype=int)

    def candidates_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return index, candidates_map_jit(rows, n_neighbors, current_graph, heap_updates[index])

    def candidates_reduce(index):
        return candidates_reduce_jit(n_tasks, current_graph, new_candidate_neighbors, old_candidate_neighbors, heap_updates, offsets, index)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
    # run map functions
    for index, count in executor.map(candidates_map, range(n_tasks)):
        heap_update_counts[index] = count

    # sort and chunk heap updates so they can be applied in the reduce
    max_count = heap_update_counts.max()
    offsets = np.zeros((n_tasks, max_count), dtype=int)

    def shuffle(index):
        return shuffle_jit(heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index)

    for _ in executor.map(shuffle, range(n_tasks)):
        pass

    # then run reduce functions
    for _ in executor.map(candidates_reduce, range(n_tasks)):
        pass

    return new_candidate_neighbors, old_candidate_neighbors

@numba.njit('i8(i8[:], i8, f4[:, :], f8[:, :, :], f8[:, :, :], f8[:, :])', nogil=True)
def nn_descent_map_jit(rows, max_candidates, data, new_candidate_neighbors, old_candidate_neighbors, heap_updates):
    count = 0
    for i in rows:
        for j in range(max_candidates):
            p = int(new_candidate_neighbors[0, i, j])
            if p < 0:
                continue
            for k in range(j, max_candidates):
                q = int(new_candidate_neighbors[0, i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
                heap_updates[count] = [p, d, q, 1]
                count += 1
                heap_updates[count] = [q, d, p, 1]
                count += 1

            for k in range(max_candidates):
                q = int(old_candidate_neighbors[0, i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
                heap_updates[count] = [p, d, q, 1]
                count += 1
                heap_updates[count] = [q, d, p, 1]
                count += 1
    return count

@numba.njit('i8(i8, f8[:, :, :], f8[:, :, :], i8[:, :], i8)', nogil=True)
def nn_decent_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index):
    c = 0
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            c += heap_push(current_graph, heap_update[0], heap_update[1], heap_update[2], heap_update[3])
    return c

def nn_descent(data, n_neighbors, rng_state, max_candidates=50,
               n_iters=10, delta=0.001, rho=0.5,
               rp_tree_init=False, leaf_array=None, verbose=False, chunk_size=4, threads=2):

    dist = distances.named_distances['euclidean']

    n_vertices = data.shape[0]
    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    current_graph = init_current_graph_threaded(data, n_neighbors, chunk_size, threads)

    for n in range(n_iters):

        (new_candidate_neighbors,
         old_candidate_neighbors) = build_candidates_threaded(current_graph,
                                                     n_vertices,
                                                     n_neighbors,
                                                     max_candidates,
                                                     rng_state, rho,
                                                     chunk_size, threads)

        # store the updates in an array
        max_heap_update_count = chunk_size * max_candidates * max_candidates * 4
        heap_updates = np.zeros((n_tasks, max_heap_update_count, 4))
        heap_update_counts = np.zeros((n_tasks,), dtype=int)

        def nn_descent_map(index):
            rows = chunk_rows(chunk_size, index, n_vertices)
            return index, nn_descent_map_jit(rows, max_candidates, data, new_candidate_neighbors, old_candidate_neighbors, heap_updates[index])

        def nn_decent_reduce(index):
            return nn_decent_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        # run map functions
        for index, count in executor.map(nn_descent_map, range(n_tasks)):
            heap_update_counts[index] = count

        # sort and chunk heap updates so they can be applied in the reduce
        max_count = heap_update_counts.max()
        offsets = np.zeros((n_tasks, max_count), dtype=int)

        def shuffle(index):
            return shuffle_jit(heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index)

        for _ in executor.map(shuffle, range(n_tasks)):
            pass

        # then run reduce functions
        c = 0
        for c_part in executor.map(nn_decent_reduce, range(n_tasks)):
            c += c_part

        if c <= delta * n_neighbors * data.shape[0]:
            break

    # TODO: parallelize
    return deheap_sort(current_graph)