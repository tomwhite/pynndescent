# To run the "beefy" experiment
#
# Run on a GCE instance
# n1-standard-64 - 240GB mem, 100GB disk
# gcloud beta compute --project=hca-scale instances create ll-knn --zone=us-east1-b --machine-type=n1-standard-64 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=218219996328-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-9-stretch-v20190213 --image-project=debian-cloud --boot-disk-size=100GB --boot-disk-type=pd-standard --boot-disk-device-name=ll-knn
#
# sudo apt-get update && sudo apt-get install -y git python3-pip
# pip3 install numba numpy scipy scikit-learn
# pip3 install git+https://github.com/tomwhite/pynndescent@benchmarks
# pip3 list
# git clone https://github.com/tomwhite/pynndescent
# cd pynndescent
# git checkout benchmarks
# python3 benchmark.py

import multiprocessing
import os
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

import pynndescent.distances as dst

from pynndescent import NNDescent
from pynndescent import pynndescent_
from pynndescent import threaded

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

np.random.seed(42)

N = 1000000
D = 128
dataset = np.random.rand(N, D).astype(np.float32)
gold = {}

n_cores = multiprocessing.cpu_count()

def cores_powers_of_two():
    i = 1
    while True:
        yield i
        i *= 2
        if i > n_cores:
            break

def scikitlearn_brute(X, threads=1, n_neighbors=25, max_candidates=50):
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    t1 = time.time()
    gold[X.shape[0]] = indices
    return indices, distances, t1-t0

def scikitlearn_ball_tree(X, threads=1, n_neighbors=25, max_candidates=50):
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    t1 = time.time()
    return indices, distances, t1-t0

def pynndescent_regular(X, threads=1, n_neighbors=25, max_candidates=50):
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    t0 = time.time()
    index = NNDescent(X, n_neighbors=n_neighbors, max_candidates=max_candidates, tree_init=False)
    indices, distances = index._neighbor_graph
    t1 = time.time()
    return indices, distances, t1-t0

def pynndescent_threaded(X, threads=1, n_neighbors=25, max_candidates=50):
    t0 = time.time()
    index = NNDescent(X, n_neighbors=n_neighbors, max_candidates=max_candidates, tree_init=False, algorithm='threaded', chunk_size=X.shape[0]//threads, threads=threads)
    indices, distances = index._neighbor_graph
    t1 = time.time()
    return indices, distances, t1-t0

def rp_tree_init_regular(X, threads=1, n_neighbors=25, max_candidates=50):
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    current_graph = pynndescent_.init_current_graph(X, dist=dst.euclidean, dist_args=(), n_neighbors=n_neighbors, rng_state=rng_state)
    t0 = time.time()
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    pynndescent_.init_rp_tree(X, dist=dst.euclidean, dist_args=(), current_graph=current_graph, n_trees=8, leaf_size=15, rng_state=rng_state)
    t1 = time.time()
    return None, None, t1-t0

def rp_tree_init_threaded(X, threads=1, n_neighbors=25, max_candidates=50):
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    current_graph = pynndescent_.init_current_graph(X, dist=dst.euclidean, dist_args=(), n_neighbors=n_neighbors, rng_state=rng_state)
    t0 = time.time()
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    threaded.init_rp_tree(X, dist=dst.euclidean, dist_args=(), current_graph=current_graph, n_trees=8, leaf_size=15, chunk_size=X.shape[0]//threads, rng_state=rng_state, threads=threads)
    t1 = time.time()
    return None, None, t1-t0

def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum() / expected.size

def all_experiments():
    n_neighbors = 25
    max_candidates = 50
    for rows in (1000, 5000, 10000, 20000, 50000, 100000):
        for threads in (1, ):
            yield (scikitlearn_brute, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000):
        for threads in (1, ):
            yield (scikitlearn_ball_tree, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000, 50000, 100000, 1000000):
        for threads in cores_powers_of_two():
            if rows < 50000 and threads > 1:
                continue
            if rows >= 1000000 and threads > 1:
                continue
            yield (pynndescent_regular, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000, 50000, 100000, 1000000):
        for threads in cores_powers_of_two():
            if rows >= 50000 and threads < 4:
                continue
            if rows >= 1000000 and threads < n_cores:
                continue
            yield (pynndescent_threaded, rows, threads, n_neighbors, max_candidates)

def generate_experiments(predicate=None):
    for exp in all_experiments():
        if predicate is None or predicate(exp):
            yield exp

# modify the predicate to run a subset of experiments
#predicate = lambda exp: (exp[0] == scikitlearn_brute or exp[0] == pynndescent_threaded) and exp[1] >= 100000 and exp[2] == 8
#predicate = lambda exp: (exp[0] == scikitlearn_brute and exp[1] == 50000) or (exp[0] == pynndescent_regular and exp[1] == 50000) or (exp[0] == pynndescent_threaded and exp[1] == 50000 and exp[2] == 8)
predicate = lambda exp: (exp[0] == scikitlearn_brute or exp[0] == pynndescent_regular or exp[0] == pynndescent_threaded) and exp[1] == 20000
# for algorithm, rows, threads, n_neighbors, max_candidates in generate_experiments(predicate):
#     indices, distances, t = algorithm(dataset[:rows], threads, n_neighbors, max_candidates)
#     acc = accuracy(gold[rows], indices) if rows in gold else -1
#     print("{},{},{},{},{},{},{}".format(algorithm.__name__, threads, rows, n_neighbors, max_candidates, t, acc))


def rep_init_experiments():
    n_neighbors = 25
    max_candidates = 50
    for rows in (1000, 5000, 10000, 20000, 50000, 100000, 1000000):
        for threads in (1,):
            yield (rp_tree_init_regular, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000, 50000, 100000, 1000000):
        for threads in cores_powers_of_two():
            if rows >= 50000 and threads < 4:
                continue
            yield (rp_tree_init_threaded, rows, threads, n_neighbors, max_candidates)

for algorithm, rows, threads, n_neighbors, max_candidates in rep_init_experiments():
    indices, distances, t = algorithm(dataset[:rows], threads, n_neighbors, max_candidates)
    acc = -1
    print("{},{},{},{},{},{},{}".format(algorithm.__name__, threads, rows, n_neighbors, max_candidates, t, acc))
