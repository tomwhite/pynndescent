from pynndescent.heap import *

def test_sparse():
    heap = make_heap(5, 2)
    heap_push(heap, 1, 6, 2, 0)
    heap_push(heap, 1, 5, 3, 0)
    heap_push(heap, 2, 5, 1, 0)

    heap_sparse = make_heap_sparse(5, 2)
    heap_push_sparse(heap_sparse, 1, 6, 2, 0)
    heap_push_sparse(heap_sparse, 1, 5, 3, 0)
    heap_push_sparse(heap_sparse, 2, 5, 1, 0)
