# To run the "beefy" experiment
#
# Run on a GCE instance
# n1-standard-64 - 240GB mem, 100GB disk
# gcloud compute --project=hca-scale instances create ll-knn --zone=us-east1-b --machine-type=n1-standard-64 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=218219996328-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-9-stretch-v20190213 --image-project=debian-cloud --boot-disk-size=100GB --boot-disk-type=pd-standard --boot-disk-device-name=ll-knn
#
# sudo apt-get update && sudo apt-get install -y git python3-pip
# pip3 install numba numpy scipy scikit-learn
# pip3 install git+https://github.com/tomwhite/pynndescent@benchmarks
# pip3 list
# git clone https://github.com/tomwhite/pynndescent
# cd pynndescent
# python3 benchmark.py

import os
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pynndescent import NNDescent
from pynndescent import threaded

np.random.seed(42)

N = 100000
D = 128
dataset = np.random.rand(N, D).astype(np.float32)

def scikitlearn_brute(X, threads=1, n_neighbors=25, max_candidates=50):
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    t1 = time.time()
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
    indices, distances = threaded.nn_descent(X, n_neighbors=n_neighbors, max_candidates=max_candidates, rng_state=None, chunk_size=X.shape[0]//threads, threads=threads)
    t1 = time.time()
    return indices, distances, t1-t0

def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum() / expected.size

def local_experiments():
    n_neighbors = 25
    max_candidates = 50
    for rows in (1000, 5000, 10000, 20000, 50000, 100000):
        for threads in (1, ):
            yield (scikitlearn_brute, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000):
        for threads in (1, ):
            yield (scikitlearn_ball_tree, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000, 50000, 100000):
        for threads in (8, ): # 4 cores * 2 (hyperthreading)
            yield (pynndescent_regular, rows, threads, n_neighbors, max_candidates)
    for rows in (1000, 5000, 10000, 20000, 50000, 100000):
        for threads in (1, 8):
            if rows >= 100000 and threads < 8:
                continue
            yield (pynndescent_threaded, rows, threads, n_neighbors, max_candidates)

def beefy_experiments():
    n_neighbors = 25
    max_candidates = 50
    for rows in (50000, 100000):
        for threads in (1, 2, 4, 8, 16, 32, 64):
            yield (pynndescent_regular, rows, threads, n_neighbors, max_candidates)
    for rows in (50000, 100000):
        for threads in (8, 16, 32, 64):
            yield (pynndescent_threaded, rows, threads, n_neighbors, max_candidates)

all_experiments = beefy_experiments

def generate_experiments(predicate=None):
    for exp in all_experiments():
        if predicate is None or predicate(exp):
            yield exp

# modify the predicate to run a subset of experiments
#predicate = lambda exp: (exp[0] == scikitlearn_brute or exp[0] == pynndescent_threaded) and exp[1] >= 100000 and exp[2] == 8
predicate = lambda exp: exp[0] == pynndescent_regular and exp[2] == 1
for algorithm, rows, threads, n_neighbors, max_candidates in generate_experiments():
    indices_sk, distances, t = algorithm(dataset[:rows], threads, n_neighbors, max_candidates)
    print("{},{},{},{},{},{}".format(algorithm.__name__, threads, rows, n_neighbors, max_candidates, t))
    # print(indices_sk)
    # print(distances)

