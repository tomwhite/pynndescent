# PyNNDescent-Spark

An implementation of PyNNDescent in PySpark.

## Motivation

PyNNDescent is used in [Scanpy] to compute nearest neighbors for UMAP. For very large
numbers of samples this can take a long time. Running a distributed version of
PyNNDescent, using Spark will help speed this up.

### Data structures

The key data structures used in NNDescent are heaps, which hold the state of the
graph of nearest neighbors, and candidate neighbors. A heap is represented as a
three-dimensional NumPy array, and is illustrated below for a graph of nearest
neighbors for 6 points and 2 nearest neighbors.

![Heap](heap.png)

Axis 0 indicates what the values stored in that slice represent: indices, weights,
or flags, corresponding to index values 0, 1, and 2.

Axis 1 corresponds to the row in the data. Axis 2 corresponds to the size of the
heap, so the number of nearest neighbors in the case shown in the diagram.

For axis 0 if the index value is 0, the values are the indices of the neighbors.
So for example, if `heap[0, i, j]` stored the value `k`, then that means that the data
point `i` has point `k` as a neighbor. The weight of the `i` to `k` link is
stored in `heap[1, i, j]`. Since it is a heap, the values are sorted along axis 2,
in ascending order of weight. The `heap[2, i, j]` values stores a flag indicating
if the candidate neighbor is new or not. This bookkeeping information is
used by the algorithm to avoid repeating work unnecessarily.

In the distributed implementation, arrays are chunked along axis 1, since this
is the number of points in the data, which may be millions or more for single cell
data.

### The NNDescent algorithm

The heart of the algorithm is implemented in `nn_descent` in `pynndescent/pynndescent_.py`.

The main steps in the algorithm at a very high level are described by this pseudo code:

```
randomly initialize current graph
loop n times
    build candidate neighbors for current graph
    update current graph with candidate neighbors
```

Each of these steps is now discussed in the context of the distributed implementation.

#### Randomly initialize current graph

The graph is seeded with random connections with random weights. If there are
`N` data points and `K` nearest neighbors then `N * K` distances are computed.

In the distributed implementation, ...

#### Build candidate neighbors for current graph

The candidate neighbors for a given point are the point's neighbors, and each of
the neighbors' neighbors.

There are two candidates heaps: one for new candidate neighbors, and one for old
candidate neighbors. (Only new neighbors need to be checked in the next step.)

In the distributed implementation, the current graph is partitioned, and each
chunk processed in its own partition. Each partition holds a temporary copy of
two heaps, for new and old candidate neighbors.

The computation is a MapReduce.
In the Map phase, the heaps are updated using information from the current graph
partition. (The local join optimization, described in [Dong], is the observation
that to find neighbors of neighbors, you don’t have to traverse `a -> b -> c`,
you just have to look at all (ordered) pairs in around `b`, which will include
`a` and `c`. In this way, the Map phase doesn't need to do a join with the rest
of the graph.)

The output of the Map is a set of `(index, candidate neighbor chunk)` pairs,
where the index is the index of the candidate neighbor chunk in the array.

These pairs are shuffled by the key (the index), and the Reduce combines all the
candidate neighbor chunks for a given index. The combine stage is simply a heap
merge.

The result is a chunked representation of the candidate neighbor heap. (Note that
the actual implementation is slightly more complicated since there are two heaps,
one for the new and one for the old candidates.)

#### Update current graph with candidate neighbors

This step involves using the candidate neighbor information to update the current
graph.

In the distributed implementation, the candidate neighbors heaps are partitioned,
and each chunk processed in its own partition. Each partition holds a temporary
copy of the current graph heap. The operation is a MapReduce, similar in form
to the one for the previous step.

### Usage

```
python3 -m venv venv  # python 3 is required!
. venv/bin/activate

pip install -e .
pip install pyspark==2.3.1 pytest
```

[Dong]: http://www.cs.princeton.edu/cass/papers/www11.pdf
[Scanpy]: https://scanpy.readthedocs.io