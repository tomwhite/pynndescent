import itertools
import numba
import numpy as np

import dask.bag as db

import pynndescent.distances as dst
from pynndescent.threaded import (
    chunk_rows,
    sort_heap_updates,
    chunk_heap_updates,
    make_current_graph_map_jit,
    candidates_map_jit,
    make_nn_descent_map_jit,
    new_rng_state,
)
from pynndescent.utils import deheap_sort, heap_push, make_heap


def listify(generator_func):
    # Turn a generator function into a function that returns a list.
    # This is needed since Dask can't pickle generators.
    def f(*args, **kwargs):
        return list(generator_func(*args, **kwargs))

    return f


def get_chunk_sizes(shape, chunks):
    def sizes(length, chunk_length):
        res = [chunk_length] * (length // chunk_length)
        if length % chunk_length != 0:
            res.append(length % chunk_length)
        return res

    return itertools.product(
        sizes(shape[0], chunks[0]),
        sizes(shape[1], chunks[1]),
        sizes(shape[2], chunks[2]),
    )


def make_heap_bag(n_points, size, chunk_size):
    shape = (3, n_points, size)
    chunks = (shape[0], chunk_size, shape[2])
    chunk_sizes = list(get_chunk_sizes(shape, chunks))

    def make_heap_chunk(chunk_size):
        return make_heap(chunk_size[1], chunk_size[2])

    return db.from_sequence(chunk_sizes, npartitions=len(chunk_sizes)).map(
        make_heap_chunk
    )


def from_bag(heap_bag):
    return np.hstack(heap_bag.compute())


@numba.njit("i8(f8[:, :, :], f4[:, :], i8)", nogil=True)
def apply_heap_updates_jit(heap, heap_updates, offset):
    c = 0
    for i in range(len(heap_updates)):
        heap_update = heap_updates[i]
        c += heap_push(
            heap,
            int(heap_update[0]) - offset,
            heap_update[1],
            int(heap_update[2]),
            int(heap_update[3]),
        )
    return c


@numba.njit("void(f8[:, :, :], f8[:, :, :], f8[:, :, :], f4[:, :], i8)", nogil=True)
def apply_new_and_old_heap_updates_jit(
    current_graph_part,
    new_candidate_neighbors,
    old_candidate_neighbors,
    heap_updates,
    offset,
):
    for i in range(len(heap_updates)):
        heap_update = heap_updates[i]
        if heap_update[3]:
            c = heap_push(
                new_candidate_neighbors,
                int(heap_update[0]) - offset,
                heap_update[1],
                int(heap_update[2]),
                int(heap_update[3]),
            )
            if c > 0:
                current_graph_part[
                    2, int(heap_update[0]) - offset, int(heap_update[4])
                ] = 0
        else:
            heap_push(
                old_candidate_neighbors,
                int(heap_update[0]) - offset,
                heap_update[1],
                int(heap_update[2]),
                int(heap_update[3]),
            )


def map_partitions_with_index(bag, func):
    indices_bag = db.from_sequence(range(bag.npartitions), npartitions=bag.npartitions)

    def funcrev(it, idx):
        return func(idx[0], it)

    return bag.map_partitions(funcrev, indices_bag)


def group_by_key(bag):
    def grouper(pair):
        return pair[0]

    def ungrouper(key_pairs_list):
        # key_pairs_list is (k, [(k, v0), (k, v1), ...])
        # return (k, [v0, v1, ...])
        key, pairs = key_pairs_list
        ret = []
        for pair in pairs:
            ret.append(pair[1])
        return key, ret

    return bag.groupby(grouper).map(ungrouper)


def init_current_graph(data, data_shape, dist, dist_args, n_neighbors, chunk_size):
    n_vertices = data_shape[0]
    current_graph_bag = make_heap_bag(n_vertices, n_neighbors, chunk_size)
    current_graph_map_jit = make_current_graph_map_jit(dist, dist_args)

    def create_heap_updates(index, iterator):
        data_local = data  # TODO: broadcast
        for _ in iterator:
            # Each part has its own heap updates for the current graph, which
            # are combined in the reduce stage.
            max_heap_update_count = chunk_size * n_neighbors * 2
            heap_updates_local = np.zeros((max_heap_update_count, 4), dtype=np.float32)
            rows = chunk_rows(chunk_size, index, n_vertices)
            count = current_graph_map_jit(
                rows,
                n_vertices,
                n_neighbors,
                data_local,
                heap_updates_local,
                new_rng_state(),  # TODO: per-work rng_state
                seed_per_row=True,
            )
            heap_updates_local = sort_heap_updates(heap_updates_local, count)
            offsets = chunk_heap_updates(
                heap_updates_local, count, n_vertices, chunk_size
            )
            # Split updates into chunks and return each chunk keyed by its index.
            for i in range(len(offsets) - 1):
                yield i, heap_updates_local[offsets[i] : offsets[i + 1]]

    def update_heap(index, iterator):
        for i in iterator:
            heap, updates = i
            for u in updates:
                offset = index * chunk_size
                apply_heap_updates_jit(heap, u, offset)
            yield heap

    # do a group by (rather than some aggregation function), since we don't combine/aggregate on the map side
    # use a partition function that ensures that partition index i goes to partition i, so we can zip with the original heap
    updates_bag = group_by_key(
        map_partitions_with_index(current_graph_bag, listify(create_heap_updates))
    ).map(lambda pair: pair[1])

    # merge the updates into the current graph
    return map_partitions_with_index(
        db.zip(current_graph_bag, updates_bag), listify(update_heap)
    )


def new_build_candidates(
    current_graph_bag,
    n_vertices,
    n_neighbors,
    max_candidates,
    chunk_size,
    rng_state,
    rho=0.5,
    seed_per_row=True,
):
    def create_heap_updates(index, iterator):
        for current_graph_part in iterator:
            # Each part has its own heap updates for the current graph, which
            # are combined in the reduce stage.
            max_heap_update_count = chunk_size * n_neighbors * 2
            heap_updates_local = np.zeros((max_heap_update_count, 5), dtype=np.float32)
            rows = chunk_rows(chunk_size, index, n_vertices)
            offset = chunk_size * index
            count = candidates_map_jit(
                rows,
                n_neighbors,
                current_graph_part,
                heap_updates_local,
                offset,
                rho,
                new_rng_state(),  # TODO: per-work rng_state
                seed_per_row=seed_per_row,
            )
            heap_updates_local = sort_heap_updates(heap_updates_local, count)
            offsets = chunk_heap_updates(
                heap_updates_local, count, n_vertices, chunk_size
            )
            # Split updates into chunks and return each chunk keyed by its index.
            for i in range(len(offsets) - 1):
                yield i, heap_updates_local[offsets[i] : offsets[i + 1]]

    def update_heaps(index, iterator):
        for i in iterator:
            current_graph_part, updates = i
            part_size = chunk_size
            if n_vertices % chunk_size != 0 and index == n_vertices // chunk_size:
                part_size = n_vertices % chunk_size
            new_candidate_neighbors_part = make_heap(part_size, max_candidates)
            old_candidate_neighbors_part = make_heap(part_size, max_candidates)
            for u in updates:
                offset = index * chunk_size
                apply_new_and_old_heap_updates_jit(
                    current_graph_part,
                    new_candidate_neighbors_part,
                    old_candidate_neighbors_part,
                    u,
                    offset,
                )
            yield new_candidate_neighbors_part, old_candidate_neighbors_part

    updates_bag = group_by_key(
        map_partitions_with_index(current_graph_bag, listify(create_heap_updates))
    ).map(lambda pair: pair[1])

    return map_partitions_with_index(
        db.zip(current_graph_bag, updates_bag), listify(update_heaps)
    )


def nn_descent(
    data,
    n_neighbors,
    rng_state,
    chunk_size,
    max_candidates=50,
    dist=dst.euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rho=0.5,
    rp_tree_init=False,
    leaf_array=None,
    verbose=False,
    seed_per_row=False,
):

    n_vertices = data.shape[0]

    current_graph_bag = init_current_graph(
        data, data.shape, dist, dist_args, n_neighbors, chunk_size
    )

    nn_descent_map_jit = make_nn_descent_map_jit(dist, dist_args)

    for n in range(n_iters):

        candidate_neighbors_combined = new_build_candidates(
            current_graph_bag,
            n_vertices,
            n_neighbors,
            max_candidates,
            chunk_size,
            rng_state,
            rho,
            seed_per_row,
        )

        def create_heap_updates(index, iterator):
            data_local = data  # TODO: broadcast
            for candidate_neighbors_combined_part in iterator:
                new_candidate_neighbors_part, old_candidate_neighbors_part = (
                    candidate_neighbors_combined_part
                )
                # Each part has its own heap updates for the current graph, which
                # are combined in the reduce stage.
                max_heap_update_count = chunk_size * max_candidates * max_candidates * 4
                heap_updates_local = np.zeros((max_heap_update_count, 4), dtype=np.float32)
                rows = chunk_rows(chunk_size, index, n_vertices)
                offset = chunk_size * index
                count = nn_descent_map_jit(
                    rows,
                    max_candidates,
                    data_local,
                    new_candidate_neighbors_part,
                    old_candidate_neighbors_part,
                    heap_updates_local,
                    offset,
                )
                heap_updates_local = sort_heap_updates(heap_updates_local, count)
                offsets = chunk_heap_updates(
                    heap_updates_local, count, n_vertices, chunk_size
                )
                # Split updates into chunks and return each chunk keyed by its index.
                for i in range(len(offsets) - 1):
                    yield i, heap_updates_local[offsets[i] : offsets[i + 1]]

        def update_heap(index, iterator):
            for i in iterator:
                heap, updates = i
                for u in updates:
                    offset = index * chunk_size
                    apply_heap_updates_jit(heap, u, offset)
                yield heap

        updates_bag = group_by_key(
            map_partitions_with_index(
                candidate_neighbors_combined, listify(create_heap_updates)
            )
        ).map(lambda pair: pair[1])

        current_graph_bag = map_partitions_with_index(
            db.zip(current_graph_bag, updates_bag), listify(update_heap)
        )

        # TODO: transfer c back from each partition and sum, in order to implement termination criterion
        # if c <= delta * n_neighbors * data.shape[0]:
        #     break

    # stack results (again, shouldn't collect result, but instead save to storage)
    current_graph = from_bag(current_graph_bag)

    return deheap_sort(current_graph)
