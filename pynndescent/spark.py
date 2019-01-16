from functools import partial
import math

from sklearn.utils import check_random_state

from pynndescent import distances
from pynndescent.utils import *

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

def read_zarr_chunk(arr, chunks, chunk_index):
    return arr[
           chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1),
           chunks[1] * chunk_index[1] : chunks[1] * (chunk_index[1] + 1),
           chunks[2] * chunk_index[2] : chunks[2] * (chunk_index[2] + 1),
           ]

def get_chunk_indices(shape, chunks):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    """
    return [
        (i, j, k)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
        for k in range(int(math.ceil(float(shape[2]) / chunks[2])))
    ]

def read_chunks(arr, chunks):
    shape = arr.shape
    func = partial(read_zarr_chunk, arr, chunks)
    chunk_indices = get_chunk_indices(shape, chunks)
    return func, chunk_indices

def to_rdd(sc, arr, chunks):
    func, chunk_indices = read_chunks(arr, chunks)
    local_rows = [func(i) for i in chunk_indices]
    return sc.parallelize(local_rows, len(local_rows))

def to_local_rows(sc, arr, chunks):
    func, chunk_indices = read_chunks(arr, chunks)
    return [func(i) for i in chunk_indices]

def init_current_graph(data, n_neighbors, random_state):
    dist = distances.named_distances['euclidean']

    random_state = check_random_state(random_state)
    rng_state = \
        random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    current_graph = make_heap(data.shape[0], n_neighbors)
    # for each row i
    for i in range(data.shape[0]):
        # choose K rows from the whole matrix
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        print(indices)
        # and work out the dist from row i to each of the random K rows
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)

    return current_graph

def build_candidates():
    pass