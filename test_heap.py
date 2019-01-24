from pynndescent.heap import *

from numpy.testing import assert_allclose

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

def test_densify():
    heap = make_heap(5, 2)
    heap_push(heap, 1, 6, 2, 0)
    heap_push(heap, 1, 5, 3, 0)
    heap_push(heap, 2, 5, 1, 0)

    heap_sparse = make_heap_sparse(5, 2)
    heap_push_sparse(heap_sparse, 1, 6, 2, 0)
    heap_push_sparse(heap_sparse, 1, 5, 3, 0)
    heap_push_sparse(heap_sparse, 2, 5, 1, 0)

    assert_allclose(densify(heap_sparse), heap)

def test_merge_heaps_dense_sparse():
    heap = make_heap(5, 2)
    heap_push(heap, 1, 6, 2, 0)
    heap_push(heap, 1, 5, 3, 0)
    heap_push(heap, 2, 5, 1, 0)

    heap_sparse = make_heap_sparse(5, 2)
    heap_push_sparse(heap_sparse, 1, 6, 2, 0)
    heap_push_sparse(heap_sparse, 1, 5, 3, 0)
    heap_push_sparse(heap_sparse, 2, 5, 1, 0)

    heap_sparse_densified = make_heap(5, 2)
    merge_heaps_dense_sparse(heap_sparse_densified, heap_sparse)

    assert_allclose(heap_sparse_densified, heap)

def test_merge_heaps_both_sparse():
    heap = make_heap_sparse(5, 2)
    heap_push_sparse(heap, 1, 4, 3, 0)
    heap_push_sparse(heap, 1, 6, 2, 0)
    heap_push_sparse(heap, 1, 5, 3, 0)
    heap_push_sparse(heap, 2, 5, 1, 0)

    heap1 = make_heap_sparse(5, 2)
    heap_push_sparse(heap1, 1, 4, 3, 0)
    heap_push_sparse(heap1, 1, 6, 2, 0)

    heap2 = make_heap_sparse(5, 2)
    heap_push_sparse(heap2, 1, 5, 3, 0)
    heap_push_sparse(heap2, 2, 5, 1, 0)

    merged = merge_heaps_sparse(heap1, heap2)

    assert_allclose(merged[0].toarray(), heap[0].toarray())
    assert_allclose(merged[1].toarray(), heap[1].toarray())
    assert_allclose(merged[2].toarray(), heap[2].toarray())
    assert merged[3] == heap[3]

def test_merge_heaps_sparse_csr():
    heap = make_heap(5, 2)
    heap_push(heap, 1, 6, 2, 0)
    heap_push(heap, 1, 5, 3, 0)
    heap_push(heap, 2, 5, 1, 0)

    heap_sparse = make_heap_sparse(5, 2)
    heap_push_sparse(heap_sparse, 1, 6, 2, 0)
    heap_push_sparse(heap_sparse, 1, 5, 3, 0)
    heap_push_sparse(heap_sparse, 2, 5, 1, 0)

    # side effect of chunking is to convert to csr
    heap_sparse_csr = next(chunk_heap_sparse(heap_sparse, (5, 2)))

    heap_sparse_densified = make_heap(5, 2)
    merge_heaps_dense_sparse(heap_sparse_densified, heap_sparse_csr)

    assert_allclose(heap_sparse_densified, heap)
