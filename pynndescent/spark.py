from functools import partial
import math

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
