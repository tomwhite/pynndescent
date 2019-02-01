import h5py
import time
import zarr

from pynndescent import NNDescent
from pynndescent import threaded
import numpy as np

from sklearn.neighbors import NearestNeighbors

d = h5py.File("/Users/tom/Downloads/sift-128-euclidean.hdf5")

test = d['test']

def time_pynndescent(X):
    t0 = time.time()
    index = NNDescent(X, n_neighbors=100, max_candidates=15, tree_init=False)
    indices, distances = index._neighbor_graph
    t1 = time.time()
    return indices, distances, t1-t0

def time_pynndescent_threaded(X, threads):
    t0 = time.time()
    indices, distances = threaded.nn_descent(X, n_neighbors=100, max_candidates=15, rng_state=None, chunk_size=X.shape[0]//threads, threads=threads)
    t1 = time.time()
    return indices, distances, t1-t0

def time_scikitlearn(X):
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    t1 = time.time()
    return indices, distances, t1-t0

def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum() / expected.size

X = test[:]
size = 20000

indices_sk, distances, t = time_scikitlearn(X[:size])
print("scikitlearn took {}s".format(t))
# print(indices_sk)
# print(distances)

for threads in (1, 2, 4, 8):
    indices, distances, t = time_pynndescent_threaded(X[:size], threads=threads)
    print("pynndescent_threaded took {}s with {} threads".format(t, threads))
    # print(indices)
    # print(distances)

print("pynndescent_threaded", accuracy(indices_sk, indices))

indices, distances, t = time_pynndescent(X[:size])
print("pynndescent took {}s".format(t))
# print(indices)
# print(distances)

print("pynndescent", accuracy(indices_sk, indices))
