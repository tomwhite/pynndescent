from pynndescent.heap import *

from numpy.testing import assert_allclose

def test_merge_heaps_sparse():
    heap = make_heap(5, 2)
    heap_push(heap, 1, 6, 2, 0)
    heap_push(heap, 1, 5, 3, 0)
    heap_push(heap, 2, 5, 1, 0)

    heap_sparse = make_heap_sparse(5, 2)
    heap_push_sparse(heap_sparse, 1, 6, 2, 0)
    heap_push_sparse(heap_sparse, 1, 5, 3, 0)
    heap_push_sparse(heap_sparse, 2, 5, 1, 0)

    heap_sparse_densified = make_heap(5, 2)
    merge_heaps_sparse(heap_sparse_densified, heap_sparse)

    assert_allclose(heap_sparse_densified, heap)

def test_merge_heaps():
    heap = make_heap(5, 2)
    heap_push(heap, 1, 4, 3, 0)
    heap_push(heap, 1, 6, 2, 0)
    heap_push(heap, 1, 5, 3, 0)
    heap_push(heap, 2, 5, 1, 0)

    heap1 = make_heap(5, 2)
    heap_push(heap1, 1, 4, 3, 0)
    heap_push(heap1, 1, 6, 2, 0)

    heap2 = make_heap(5, 2)
    heap_push(heap2, 1, 5, 3, 0)
    heap_push(heap2, 2, 5, 1, 0)

    assert_allclose(merge_heaps(heap1, heap2), heap)


# def test():
#     heap_sparse = make_heap_sparse(5, 2)
#     heap_push_sparse(heap_sparse, 1, 6, 2, 0)
#     heap_push_sparse(heap_sparse, 1, 5, 3, 0)
#     heap_push_sparse(heap_sparse, 2, 5, 1, 0)
#
#     print(heap_sparse)
#     print(heap_sparse[1].toarray())
#     csr = heap_sparse[1].tocsr()
#     print(csr.data)
#     print(csr.indices)
#     print(csr.indptr)
#
#     print([i for i in range(len(csr.indptr) - 1) if csr.indptr[i] != csr.indptr[i+1]])
#     print(heap_sparse[0])
#
#     print(sparse_to_chunks(heap_sparse, (3, 2)))