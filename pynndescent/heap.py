import itertools
import math
import numpy as np
import scipy.sparse

from pynndescent import utils

# support dense and sparse heaps
def make_heap(n_points, size, sparse=False):
    if sparse:
        return make_heap_sparse(n_points, size)
    return utils.make_heap(n_points, size)

def heap_push(heap, row, weight, index, flag):
    if isinstance(heap, tuple):
        return heap_push_sparse(heap, row, weight, index, flag)
    return utils.heap_push(heap, row, weight, index, flag)

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

def from_rdd(heap_rdd):
    return np.hstack(heap_rdd.collect())

# Sparse heap functions

def make_heap_sparse(n_points, size):
    # scipy.sparse only supports 2D matrices, so we have one for each of the
    # three positions in the first axis.
    # Even if we used pydata.sparse arrays (which support 3D) we couldn't use
    # fill values since each slice has different fill values.
    # Note that we need to keep track of populated rows in order to do filling,
    # but also to do efficient merging of sparse arrays.
    # TODO: write own sparse array (3D too?) that is row-oriented, but only stores populated rows - Dictionary Of Rows
    indices = scipy.sparse.dok_matrix((n_points, size))
    weights = scipy.sparse.dok_matrix((n_points, size))
    is_new = scipy.sparse.dok_matrix((n_points, size))
    rows = set() # which rows have been created
    return indices, weights, is_new, rows

def print_heap_sparse(heap):
    print("indices", heap[0].toarray())
    print("weights", heap[1].toarray())
    print("is_new", heap[2].toarray())
    print("rows", heap[3])

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
    indices = heap[0]
    weights = heap[1]
    is_new = heap[2]
    rows = heap[3]

    if row not in rows:
        # initialize
        indices[row] = -1
        weights[row] = np.infty
        is_new[row] = 0
        rows.add(row)

    if weight >= weights[row,0]:
        return 0

    # break if we already have this element.
    for i in range(indices.shape[1]):
        if index == indices[row,i]:
            return 0

    # insert val at position zero
    weights[row,0] = weight
    indices[row,0] = index
    is_new[row,0] = flag

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= indices.shape[1]:
            break
        elif ic2 >= indices.shape[1]:
            if weights[row,ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[row,ic1] >= weights[row,ic2]:
            if weight < weights[row,ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[row,ic2]:
                i_swap = ic2
            else:
                break

        weights[row,i] = weights[row,i_swap]
        indices[row,i] = indices[row,i_swap]
        is_new[row,i] = is_new[row,i_swap]

        i = i_swap

    weights[row,i] = weight
    indices[row,i] = index
    is_new[row,i] = flag

    return 1

def rows_in_sparse(heap_sparse):
    if len(heap_sparse) == 4: # row information is last item in tuple
        return heap_sparse[3]
    else: # CSR format where row information is implicit
        indptr = heap_sparse[0].indptr
        return [i for i in range(len(indptr) - 1) if indptr[i] != indptr[i+1]]

def from_rdd_sparse(heap_sparse_rdd):
    heap_chunks = heap_sparse_rdd.collect()
    chunk_size = heap_chunks[0][0].shape[0]
    rows = set()
    for i, heap_chunk in enumerate(heap_chunks):
        row_start = i * chunk_size
        rows.update([int(row + row_start) for row in heap_chunk[3]])
    weights = scipy.sparse.vstack([heap_chunk[0] for heap_chunk in heap_chunks], format="dok")
    indices = scipy.sparse.vstack([heap_chunk[1] for heap_chunk in heap_chunks], format="dok")
    is_new = scipy.sparse.vstack([heap_chunk[2] for heap_chunk in heap_chunks], format="dok")
    return weights, indices, is_new, rows

def densify(heap_sparse):
    shape = heap_sparse[0].shape
    heap = make_heap(shape[0], shape[1])
    for i in rows_in_sparse(heap_sparse):
        for j in range(shape[1]):
            heap[0, i, j] = heap_sparse[0][i, j]
            heap[1, i, j] = heap_sparse[1][i, j]
            heap[2, i, j] = heap_sparse[2][i, j]
    return heap

def densify_pair(pair):
    return densify(pair[0]), densify(pair[1])

def merge_heaps_dense_sparse(heap1_dense, heap2_sparse):
    # TODO: check heaps have the same size
    all_indices = heap2_sparse[0]
    all_weights = heap2_sparse[1]
    all_is_new = heap2_sparse[2]
    for row in rows_in_sparse(heap2_sparse):
        for ind in range(all_indices.shape[1]):
            index = all_indices[row, ind]
            weight = all_weights[row, ind]
            flag = all_is_new[row, ind]
            heap_push(heap1_dense, row, weight, index, flag)
    return heap1_dense

def merge_heaps_sparse(heap1_sparse, heap2_sparse):
    # TODO: check heaps have the same size
    all_indices = heap2_sparse[0]
    all_weights = heap2_sparse[1]
    all_is_new = heap2_sparse[2]
    for row in rows_in_sparse(heap2_sparse):
        for ind in range(all_indices.shape[1]):
            index = all_indices[row, ind]
            weight = all_weights[row, ind]
            flag = all_is_new[row, ind]
            heap_push_sparse(heap1_sparse, row, weight, index, flag)
    return heap1_sparse

def merge_heap_pairs_sparse(heap_pair1_sparse, heap_pair2_sparse):
    heap1_new_sparse, heap1_old_sparse = heap_pair1_sparse
    heap2_new_sparse, heap2_old_sparse = heap_pair2_sparse
    return merge_heaps_sparse(heap1_new_sparse, heap2_new_sparse), merge_heaps_sparse(heap1_old_sparse, heap2_old_sparse)

def chunk_heap_sparse(heap_sparse, chunks):
    # converts to CSR as a side-effect, which keeps row information
    shape = heap_sparse[0].shape
    chunk_indices = [
        (i, j)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
    ]
    for chunk_index in chunk_indices:
        yield tuple(heap_sparse[i][chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1)].tocsr() for i in (0, 1, 2))

def read_heap_chunks_sparse(heap_sparse, chunks_3d):
    # TODO: refactor to reduce duplication with/similarity to chunk_heap_sparse
    shape = heap_sparse[0].shape
    chunks = chunks_3d[1:]
    def func(chunk_index):
        row_start = chunks[0] * chunk_index[0]
        row_end = chunks[0] * (chunk_index[0] + 1)
        rows = set([int(row - row_start) for row in heap_sparse[3] if row_start <= row < row_end])
        return tuple(heap_sparse[i][chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1)] for i in (0, 1, 2)) + (rows,)
    chunk_indices = [
        (i, j)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
    ]
    return func, chunk_indices
