import itertools
import math
import numpy as np
import scipy.sparse

from pynndescent.utils import *

# Distributed heap functions
# Distributed heaps can only be created and merged. Heap push is a local
# operation only.

def get_chunk_sizes(shape, chunks):
    def sizes(length, chunk_length):
        res = [chunk_length] * (length // chunk_length)
        if length % chunk_length != 0:
            res.append(length % chunk_length)
        return res

    return itertools.product(sizes(shape[0], chunks[0]), sizes(shape[1], chunks[1]), sizes(shape[2], chunks[2]))

def make_heap_rdd(sc, n_points, size, chunk_size):
    shape = (3, n_points, size)
    chunks = (shape[0], chunk_size, shape[2])
    chunk_sizes = list(get_chunk_sizes(shape, chunks))
    def make_heap_chunk(chunk_size):
        return make_heap(chunk_size[1], chunk_size[2])
    return sc.parallelize(chunk_sizes, len(chunk_sizes)).map(make_heap_chunk)

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

# Sparse heap functions

def make_heap_sparse(n_points, size):
    # scipy.sparse only supports 2D matrices, so we have one for each of the
    # three positions in the first axis
    rows = set() # which rows have been created
    indices = scipy.sparse.lil_matrix((n_points, size))
    weights = scipy.sparse.lil_matrix((n_points, size))
    is_new = scipy.sparse.lil_matrix((n_points, size))
    return rows, indices, weights, is_new

def heap_push_sparse(heap, row, weight, index, flag):
    """Push a new element onto the heap. The heap stores potential neighbors
    for each data point. The ``row`` parameter determines which data point we
    are addressing, the ``weight`` determines the distance (for heap sorting),
    the ``index`` is the element to add, and the flag determines whether this
    is to be considered a new addition.

    Parameters
    ----------
    heap: ndarray generated by ``make_heap``
        The heap object to push into

    row: int
        Which actual heap within the heap object to push to

    weight: float
        The priority value of the element to push onto the heap

    index: int
        The actual value to be pushed

    flag: int
        Whether to flag the newly added element or not.

    Returns
    -------
    success: The number of new elements successfully pushed into the heap.
    """
    rows = heap[0]
    all_indices = heap[1]
    all_weights = heap[2]
    all_is_new = heap[3]

    if row not in rows:
        # initialize
        all_indices[row] = -1
        all_weights[row] = np.infty
        all_is_new[row] = 0
        rows.add(row)

    indices = all_indices.getrowview(row)
    weights = all_weights.getrowview(row)
    is_new = all_is_new.getrowview(row)

    if weight >= weights[0,0]:
        return 0

    # break if we already have this element.
    for i in range(indices.shape[1]):
        if index == indices[0,i]:
            return 0

    # insert val at position zero
    weights[0,0] = weight
    indices[0,0] = index
    is_new[0,0] = flag

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= all_indices.shape[1]:
            break
        elif ic2 >= all_indices.shape[1]:
            if weights[0,ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[0,ic1] >= weights[0,ic2]:
            if weight < weights[0,ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[0,ic2]:
                i_swap = ic2
            else:
                break

        weights[0,i] = weights[0,i_swap]
        indices[0,i] = indices[0,i_swap]
        is_new[0,i] = is_new[0,i_swap]

        i = i_swap

    weights[0,i] = weight
    indices[0,i] = index
    is_new[0,i] = flag

    return 1

def densify(heap):
    return np.stack([heap[i].toarray() for i in (1, 2, 3)])

def densify0(heap):
    return np.stack([heap[i].toarray() for i in (0, 1, 2)])

# def read_heap_chunks_sparse(heap, chunks):
#     shape = heap[1].shape
#     def func(chunk_index):
#         return densify0(tuple(heap[i][chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 0)] for i in (1, 2, 3)))
#     chunk_indices = [
#         (i, j)
#         for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
#         for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
#     ]
#     return func, chunk_indices
#
def merge_heaps_sparse(heap1_dense, heap2_sparse):
    # TODO: check heaps have the same size
    s = heap2_sparse[1].shape
    for row in heap2_sparse[0]:
        for ind in range(s[1]):
            index = heap2_sparse[1][row, ind]
            weight = heap2_sparse[2][row, ind]
            flag = heap2_sparse[3][row, ind]
            heap_push(heap1_dense, row, weight, index, flag)
    return heap1_dense

def sparse_to_chunks(heap_sparse, chunks):
    shape = heap_sparse[1].shape
    chunk_indices = [
        (i, j)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
    ]
    print(chunk_indices)

    for chunk_index in chunk_indices:
        x = tuple(heap_sparse[i][chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 0)].tocsr() for i in (1, 2, 3))
        for i in (0, 1, 2):
            print(x[i].data)
            print(x[i].indices)
            print(x[i].indptr)
