import math
import numpy as np

from numpy.testing import assert_allclose

from pynndescent import distances
from pynndescent import pynndescent_
from pynndescent import threaded
from pynndescent import utils

def new_rng_state():
    return np.empty((3,), dtype=np.int64)

def test_init_current_graph():
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32)
    n_neighbors = 2

    current_graph = threaded.init_current_graph(data, n_neighbors)
    current_graph_threaded = threaded.init_current_graph_threaded(data, n_neighbors, chunk_size=4)

    assert_allclose(current_graph_threaded, current_graph)

def test_build_candidates():
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32)
    n_vertices = data.shape[0]
    n_neighbors = 2
    max_candidates = 8

    current_graph = threaded.init_current_graph(data, n_neighbors)
    new_candidate_neighbors, old_candidate_neighbors = \
        utils.build_candidates(current_graph, n_vertices, n_neighbors, max_candidates, rng_state=new_rng_state())

    current_graph = threaded.init_current_graph(data, n_neighbors)
    new_candidate_neighbors_threaded, old_candidate_neighbors_threaded = \
        threaded.build_candidates_threaded(current_graph, n_vertices, n_neighbors, max_candidates, chunk_size=4, rng_state=new_rng_state())

    assert_allclose(new_candidate_neighbors_threaded, new_candidate_neighbors)
    assert_allclose(old_candidate_neighbors_threaded, old_candidate_neighbors)

def test_nn_descent():
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32)
    n_neighbors = 2
    max_candidates = 8

    nn_descent = pynndescent_.make_nn_descent(distances.named_distances['euclidean'], ())
    nn = nn_descent(data, n_neighbors=n_neighbors, rng_state=new_rng_state(), max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)

    nn_threaded = threaded.nn_descent(data, n_neighbors=n_neighbors, rng_state=new_rng_state(), chunk_size=4, max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)

    assert_allclose(nn, nn_threaded)

def test_heap_updates():
    heap_updates = np.array([
        [4, 1, 15, 0],
        [3, 3, 12, 0],
        [2, 2, 14, 0],
        [1, 5, 29, 0],
        [4, 7, 40, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64)
    num_heap_updates = 5
    chunk_size = 2
    sorted_heap_updates = threaded.sort_heap_updates(heap_updates, num_heap_updates)
    offsets = threaded.chunk_heap_updates(sorted_heap_updates, 6, chunk_size)

    assert_allclose(offsets, np.array([0, 1, 3, 5]))

    chunk0 = sorted_heap_updates[offsets[0]:offsets[1]]
    assert_allclose(chunk0, np.array([
        [1, 5, 29, 0]
    ]))

    chunk1 = sorted_heap_updates[offsets[1]:offsets[2]]
    assert_allclose(chunk1, np.array([
        [2, 2, 14, 0],
        [3, 3, 12, 0]
    ]))

    chunk2 = sorted_heap_updates[offsets[2]:offsets[3]]
    assert_allclose(chunk2, np.array([
        [4, 1, 15, 0],
        [4, 7, 40, 0]
    ]))
