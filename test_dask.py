import numpy as np

from numpy.testing import assert_allclose

import dask
import dask.bag as db

from sklearn.neighbors import NearestNeighbors

import pynndescent
from pynndescent.dask import make_heap_bag, from_bag, map_partitions_with_index, group_by_key
from pynndescent import distances
from pynndescent import NNDescent
from pynndescent import pynndescent_
from pynndescent import utils

# run tests single-threaded to get predictable ordering (e.g. with groupby)
dask.config.set(scheduler='single-threaded')

data = np.array(
    [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32
)
chunk_size = 4
n_neighbors = 2
max_candidates = 8

dist = distances.named_distances["euclidean"]
dist_args = ()

def new_rng_state():
    return np.empty((3,), dtype=np.int64)


def test_convenience_functions():
    bag = db.from_sequence([1, 2, 3, 4, 5, 6], npartitions=3)
    def fn(idx, it):
        return [(2 - idx, val) for val in it]
    x = map_partitions_with_index(bag, fn)
    assert x.compute() == [(2, 1), (2, 2), (1, 3), (1, 4), (0, 5), (0, 6)]

    y = group_by_key(x)
    assert y.compute() == [(0, [5, 6]), (1, [3, 4]), (2, [1, 2])]


def test_init_current_graph():
    current_graph = pynndescent_.init_current_graph(data, dist, dist_args, n_neighbors, rng_state=new_rng_state(), seed_per_row=True)
    current_graph_dask = pynndescent.dask.init_current_graph_bag(data, data.shape, dist, dist_args, n_neighbors, chunk_size=4)

    current_graph_dask_materialized = from_bag(current_graph_dask)

    assert_allclose(current_graph_dask_materialized, current_graph)


def test_build_candidates():
    n_vertices = data.shape[0]

    current_graph = pynndescent_.init_current_graph(data, dist, dist_args, n_neighbors, rng_state=new_rng_state(), seed_per_row=True)
    new_candidate_neighbors, old_candidate_neighbors = utils.build_candidates(
        current_graph,
        n_vertices,
        n_neighbors,
        max_candidates,
        rng_state=new_rng_state(),
        seed_per_row=True
    )

    current_graph_dask = pynndescent.dask.init_current_graph_bag(data, data.shape, dist, dist_args, n_neighbors, chunk_size=4)
    candidate_neighbors_combined_dask = pynndescent.dask.build_candidates_bag(
        current_graph_dask,
        n_vertices,
        n_neighbors,
        max_candidates,
        chunk_size=chunk_size,
        rng_state=new_rng_state(),
        seed_per_row=True
    )

    candidate_neighbors_combined = candidate_neighbors_combined_dask.compute()
    new_candidate_neighbors_dask = np.hstack(
        [pair[0] for pair in candidate_neighbors_combined]
    )
    old_candidate_neighbors_dask = np.hstack(
        [pair[1] for pair in candidate_neighbors_combined]
    )

    assert_allclose(new_candidate_neighbors_dask, new_candidate_neighbors)
    assert_allclose(old_candidate_neighbors_dask, old_candidate_neighbors)

def test_nn_descent():
    nn_indices, nn_distances = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=2,
        delta=0,
        tree_init=False,
        seed_per_row=True
    )._neighbor_graph

    nn_indices_threaded, nn_distances_threaded = NNDescent(
        data,
        n_neighbors=n_neighbors,
        max_candidates=max_candidates,
        n_iters=2,
        delta=0,
        tree_init=False,
        seed_per_row=True,
        algorithm='dask',
        chunk_size=chunk_size
    )._neighbor_graph

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(data)
    _, nn_gold_indices = nbrs.kneighbors(data)

    assert_allclose(nn_indices_threaded, nn_indices)
    assert_allclose(nn_distances_threaded, nn_distances)