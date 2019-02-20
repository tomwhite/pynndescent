import h5py
import time
import zarr

from pynndescent import NNDescent
from pynndescent import threaded
import numpy as np

from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

d = h5py.File("/Users/tom/Downloads/sift-128-euclidean.hdf5")

test = d['test']

N = 100000
D = 128

dataset = np.random.rand(N, D).astype(np.float)

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

X = dataset
size = 20000

def generate_experiments():
    # for rows in (1000, 5000, 10000, 20000, 50000):
    #     for threads in (1, ):
    #         yield (scikitlearn_brute, rows, threads)
    # for rows in (1000, 5000, 10000, 20000):
    #     for threads in (1, ):
    #         yield (scikitlearn_ball_tree, rows, threads)
    for rows in (1000, 5000, 10000, 20000):
        for threads in (8, ): # 4 cores * 2 (hyperthreading)
            yield (pynndescent_regular, rows, threads)
    for rows in (1000, 5000, 10000, 20000):
        for threads in (1, 8):
            yield (pynndescent_threaded, rows, threads)

for algorithm, rows, threads in generate_experiments():
    indices_sk, distances, t = algorithm(X[:rows], threads)
    print("{},{},{},{}".format(algorithm.__name__, threads, rows, t))
    # print(indices_sk)
    # print(distances)

# algorithm=scikitlearn_ball_tree
# for rows in (1000, 5000, 10000, 20000, 50000):
#     for threads in (1, ):
#         indices_sk, distances, t = algorithm(X[:rows], threads)
#         print("{},{},{},{}".format(algorithm.__name__, threads, rows, t))
#         # print(indices_sk)
#         # print(distances)
#
# for threads in (1, 2, 4, 8):
#     indices, distances, t = time_pynndescent_threaded(X[:size], threads)
#     print("pynndescent_threaded took {}s with {} threads".format(t, threads))
#     # print(indices)
#     # print(distances)
#
# print("pynndescent_threaded", accuracy(indices_sk, indices))
#
# indices, distances, t = time_pynndescent(X[:size])
# print("pynndescent took {}s".format(t))
# # print(indices)
# # print(distances)
#
# print("pynndescent", accuracy(indices_sk, indices))
