import concurrent.futures
import math
import numba
import numpy as np

from pynndescent import distances
from pynndescent.utils import deheap_sort, heap_push, make_heap, rejection_sample, seed, tau_rand

# NNDescent algorithm

@numba.njit('i8[:](i8, i8, i8)', nogil=True)
def chunk_rows(chunk_size, index, n_vertices):
    return np.arange(chunk_size * index, min(chunk_size * (index + 1), n_vertices))

@numba.njit('f8[:, :](f8[:, :], i8)', nogil=True)
def sort_heap_updates(heap_updates, num_heap_updates):
    """Take an array of unsorted heap updates and sort by row number."""
    row_numbers = heap_updates[:num_heap_updates, 0]
    return heap_updates[:num_heap_updates][row_numbers.argsort()]

@numba.njit('i8[:](f8[:, :], i8, i8)', nogil=True)
def chunk_heap_updates(heap_updates, n_vertices, chunk_size):
    """Return the offsets for each chunk of sorted heap updates."""
    chunk_boundaries = np.arange(int(math.ceil(float(n_vertices) / chunk_size)) + 1) * chunk_size
    offsets = np.searchsorted(heap_updates[:, 0], chunk_boundaries, side='left')
    return offsets

@numba.njit('void(f8[:, :, :], i8[:], i8[:, :], i8, i8, i8)', nogil=True)
def shuffle_jit(heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index):
    sorted_heap_updates = sort_heap_updates(heap_updates[index], heap_update_counts[index])
    o = chunk_heap_updates(sorted_heap_updates, n_vertices, chunk_size)
    offsets[index, :o.shape[0]] = o

@numba.njit('i8(f8[:, :, :], f8[:, :], i8)', nogil=True)
def apply_heap_updates_jit(heap, heap_updates, offset):
    c = 0
    for i in range(len(heap_updates)):
        heap_update = heap_updates[i]
        c += heap_push(heap, int(heap_update[0]) - offset, heap_update[1], int(heap_update[2]), int(heap_update[3]))
    return c

@numba.njit('void(f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :], i8)', nogil=True)
def apply_new_and_old_heap_updates_jit(current_graph_part, new_candidate_neighbors, old_candidate_neighbors, heap_updates, offset):
    for i in range(len(heap_updates)):
        heap_update = heap_updates[i]
        if heap_update[3]:
            c = heap_push(new_candidate_neighbors, int(heap_update[0]) - offset, heap_update[1], int(heap_update[2]), int(heap_update[3]))
            if c > 0:
                current_graph_part[2, int(heap_update[0]) - offset, int(heap_update[4])] = 0
        else:
            heap_push(old_candidate_neighbors, int(heap_update[0]) - offset, heap_update[1], int(heap_update[2]), int(heap_update[3]))

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

def init_current_graph_threaded(data, n_neighbors, chunk_size, threads=2):

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

@numba.njit('i8(i8[:], i8, f8[:, :, :], f8[:, :], i8)', nogil=True)
def candidates_map_jit(rows, n_neighbors, current_graph, heap_updates, offset):
    rng_state = np.empty((3,), dtype=np.int64)
    count = 0
    for i in rows:
        seed(rng_state, i)
        for j in range(n_neighbors):
            if current_graph[0, i - offset, j] < 0:
                continue
            idx = current_graph[0, i - offset, j]
            isn = current_graph[2, i - offset, j]
            d = tau_rand(rng_state)
            if tau_rand(rng_state) < rho:
                # updates are common to old and new - decided by 'isn' flag
                hu = heap_updates[count]
                hu[0] = i
                hu[1] = d
                hu[2] = idx
                hu[3] = isn
                hu[4] = j
                count += 1
                hu = heap_updates[count]
                hu[0] = idx
                hu[1] = d
                hu[3] = isn
                hu[4] = j
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

def build_candidates_threaded(current_graph, n_vertices, n_neighbors, max_candidates, chunk_size,
                         rng_state, rho=0.5, threads=2):

    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    new_candidate_neighbors = make_heap(n_vertices, max_candidates)
    old_candidate_neighbors = make_heap(n_vertices, max_candidates)

    # store the updates in an array
    max_heap_update_count = chunk_size * n_neighbors * 2
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 5))
    heap_update_counts = np.zeros((n_tasks,), dtype=int)

    def candidates_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return index, candidates_map_jit(rows, n_neighbors, current_graph, heap_updates[index], offset=0)

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

@numba.njit('i8(i8[:], i8, f4[:, :], f8[:, :, :], f8[:, :, :], f8[:, :], i8)', nogil=True, fastmath=True)
def nn_descent_map_jit(rows, max_candidates, data, new_candidate_neighbors, old_candidate_neighbors, heap_updates, offset):
    count = 0
    for i in rows:
        i -= offset
        for j in range(max_candidates):
            p = int(new_candidate_neighbors[0, i, j])
            if p < 0:
                continue
            for k in range(j, max_candidates):
                q = int(new_candidate_neighbors[0, i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
                hu = heap_updates[count]
                hu[0] = p
                hu[1] = d
                hu[2] = q
                hu[3] = 1
                count += 1
                hu = heap_updates[count]
                hu[0] = q
                hu[1] = d
                hu[2] = p
                hu[3] = 1
                count += 1

            for k in range(max_candidates):
                q = int(old_candidate_neighbors[0, i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
                hu = heap_updates[count]
                hu[0] = p
                hu[1] = d
                hu[2] = q
                hu[3] = 1
                count += 1
                hu = heap_updates[count]
                hu[0] = q
                hu[1] = d
                hu[2] = p
                hu[3] = 1
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

def nn_descent(data, n_neighbors, rng_state, chunk_size, max_candidates=50,
               n_iters=10, delta=0.001, rho=0.5,
               rp_tree_init=False, leaf_array=None, verbose=False, threads=2):

    dist = distances.named_distances['euclidean']

    n_vertices = data.shape[0]
    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    current_graph = init_current_graph_threaded(data, n_neighbors, chunk_size, threads)

    # store the updates in an array
    max_heap_update_count = chunk_size * max_candidates * max_candidates * 4
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 4))
    heap_update_counts = np.zeros((n_tasks,), dtype=int)

    for n in range(n_iters):
        import time
        t0 = time.time()
        (new_candidate_neighbors,
         old_candidate_neighbors) = build_candidates_threaded(current_graph,
                                                     n_vertices,
                                                     n_neighbors,
                                                     max_candidates,
                                                     chunk_size,
                                                     rng_state, rho,
                                                     threads)
        t1 = time.time()
        print('cand', (t1-t0))
        def nn_descent_map(index):
            rows = chunk_rows(chunk_size, index, n_vertices)
            return index, nn_descent_map_jit(rows, max_candidates, data, new_candidate_neighbors, old_candidate_neighbors, heap_updates[index], offset=0)

        def nn_decent_reduce(index):
            return nn_decent_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        # run map functions
        for index, count in executor.map(nn_descent_map, range(n_tasks)):
            heap_update_counts[index] = count

        t2 = time.time()
        print('map', (t2-t1))

        # sort and chunk heap updates so they can be applied in the reduce
        max_count = heap_update_counts.max()
        offsets = np.zeros((n_tasks, max_count), dtype=int)

        def shuffle(index):
            return shuffle_jit(heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index)

        for _ in executor.map(shuffle, range(n_tasks)):
            pass
        t3 = time.time()
        print('shuffle', (t3-t2))

        # then run reduce functions
        c = 0
        for c_part in executor.map(nn_decent_reduce, range(n_tasks)):
            c += c_part

        t4 = time.time()
        print('red', (t4-t3))
        if c <= delta * n_neighbors * data.shape[0]:
            break

    # TODO: parallelize
    return deheap_sort(current_graph)