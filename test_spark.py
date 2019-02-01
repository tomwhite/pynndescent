import logging
import numpy as np
import unittest

from numpy.testing import assert_allclose

from pyspark.sql import SparkSession

from pynndescent import distances
from pynndescent.heap import *
from pynndescent import pynndescent_
from pynndescent import utils
from pynndescent import spark

def new_rng_state():
    return np.empty((3,), dtype=np.int64)

class TestSpark(unittest.TestCase):

    # based on https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
    @classmethod
    def suppress_py4j_logging(cls):
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (
            SparkSession.builder.master("local[1]")
            .appName("my-local-testing-pyspark-context")
            .getOrCreate()
        )

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()
        cls.sc = cls.spark.sparkContext

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_init_current_graph(self):
        data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        data_broadcast = self.sc.broadcast(data)
        n_neighbors = 2

        current_graph = spark.init_current_graph(data, n_neighbors)
        current_graph_rdd = spark.init_current_graph_rdd(self.sc, data_broadcast, data.shape, n_neighbors)

        current_graph_rdd_materialized = from_rdd(current_graph_rdd)

        assert_allclose(current_graph_rdd_materialized, current_graph)

    def test_build_candidates(self):
        data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        data_broadcast = self.sc.broadcast(data)
        n_vertices = data.shape[0]
        n_neighbors = 2
        max_candidates = 8

        current_graph = spark.init_current_graph(data, n_neighbors)
        new_candidate_neighbors, old_candidate_neighbors =\
            utils.build_candidates(current_graph, n_vertices, n_neighbors, max_candidates, rng_state=new_rng_state())

        current_graph_rdd = spark.init_current_graph_rdd(self.sc, data_broadcast, data.shape, n_neighbors)
        candidate_neighbors_combined_rdd = \
            spark.build_candidates_rdd(current_graph_rdd, n_vertices, n_neighbors, max_candidates, rng_state=new_rng_state())

        candidate_neighbors_combined = candidate_neighbors_combined_rdd.collect()
        new_candidate_neighbors_spark = np.hstack([pair[0] for pair in candidate_neighbors_combined])
        old_candidate_neighbors_spark = np.hstack([pair[1] for pair in candidate_neighbors_combined])

        assert_allclose(new_candidate_neighbors_spark, new_candidate_neighbors)
        assert_allclose(old_candidate_neighbors_spark, old_candidate_neighbors)

    def test_nn_descent(self):
        data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        n_neighbors = 2
        max_candidates = 8

        nn_descent = pynndescent_.make_nn_descent(distances.named_distances['euclidean'], ())
        nn = nn_descent(data, n_neighbors=n_neighbors, rng_state=new_rng_state(), max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)

        nn_spark = spark.nn_descent(self.sc, data, n_neighbors=n_neighbors, rng_state=new_rng_state(), max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)

        assert_allclose(nn, nn_spark)
