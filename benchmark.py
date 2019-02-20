import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pynndescent import NNDescent
from pynndescent import threaded

np.random.seed(42)

N = 100000
D = 128
dataset = np.random.rand(N, D).astype(np.float32)

def scikitlearn_brute(X, threads=1):
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    t1 = time.time()
    return indices, distances, t1-t0

def scikitlearn_ball_tree(X, threads=1):
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    t1 = time.time()
    return indices, distances, t1-t0

def pynndescent_regular(X, threads=1):
    t0 = time.time()
    index = NNDescent(X, n_neighbors=100, max_candidates=15, tree_init=False)
    indices, distances = index._neighbor_graph
    t1 = time.time()
    return indices, distances, t1-t0

def pynndescent_threaded(X, threads=1):
    t0 = time.time()
    indices, distances = threaded.nn_descent(X, n_neighbors=100, max_candidates=15, rng_state=None, chunk_size=X.shape[0]//threads, threads=threads)
    t1 = time.time()
    return indices, distances, t1-t0

def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum() / expected.size

def all_experiments():
    for rows in (1000, 5000, 10000, 20000, 50000, 100000):
        for threads in (1, ):
            yield (scikitlearn_brute, rows, threads)
    for rows in (1000, 5000, 10000, 20000):
        for threads in (1, ):
            yield (scikitlearn_ball_tree, rows, threads)
    for rows in (1000, 5000, 10000, 20000, 50000):
        for threads in (8, ): # 4 cores * 2 (hyperthreading)
            yield (pynndescent_regular, rows, threads)
    for rows in (1000, 5000, 10000, 20000, 50000, 100000):
        for threads in (1, 8):
            if rows >= 100000 and threads < 8:
                continue
            yield (pynndescent_threaded, rows, threads)

def generate_experiments(predicate=None):
    for exp in all_experiments():
        if predicate is None or predicate(exp):
            yield exp

# modify the predicate to run a subset of experiments
predicate = lambda exp: (exp[0] == scikitlearn_brute or exp[0] == pynndescent_threaded) and exp[1] >= 100000 and exp[2] == 8
for algorithm, rows, threads in generate_experiments(predicate):
    indices_sk, distances, t = algorithm(dataset[:rows], threads)
    print("{},{},{},{}".format(algorithm.__name__, threads, rows, t))
    # print(indices_sk)
    # print(distances)

