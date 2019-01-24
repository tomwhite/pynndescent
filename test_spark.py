import logging
import numpy as np
import unittest

from numpy.testing import assert_allclose

from pyspark.sql import SparkSession

from pynndescent import distances
from pynndescent.heap import print_heap_sparse
from pynndescent import pynndescent_
from pynndescent import utils
from pynndescent import spark

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
        n_neighbors = 2

        current_graph = spark.init_current_graph(data, n_neighbors, spark.get_rng_state(42))
        current_graph_rdd = spark.init_current_graph_rdd(self.sc, data, n_neighbors, spark.get_rng_state(42))

        print("current graph", current_graph)

        for cg in current_graph_rdd.collect():
            print_heap_sparse(cg)

        # current_graph_rdd_materialized = np.hstack(current_graph_rdd.collect())
        #
        # assert_allclose(current_graph_rdd_materialized, current_graph)

    # def test_build_candidates(self):
    #     data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #     n_vertices = data.shape[0]
    #     n_neighbors = 2
    #     max_candidates = 8
    #
    #     current_graph = spark.init_current_graph(data, n_neighbors, spark.get_rng_state(42))
    #     new_candidate_neighbors, old_candidate_neighbors =\
    #         utils.build_candidates(current_graph, n_vertices, n_neighbors, max_candidates, spark.get_rng_state(42))
    #
    #     current_graph_rdd = spark.init_current_graph_rdd(self.sc, data, n_neighbors, spark.get_rng_state(42))
    #     new_candidate_neighbors_spark, old_candidate_neighbors_spark =\
    #         spark.build_candidates(self.sc, current_graph_rdd, n_vertices, n_neighbors, max_candidates, spark.get_rng_state(42))
    #
    #     assert_allclose(new_candidate_neighbors_spark, new_candidate_neighbors)
    #     assert_allclose(old_candidate_neighbors_spark, old_candidate_neighbors)
    #
    # def test_nn_descent(self):
    #     data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #     n_neighbors = 2
    #     max_candidates = 8
    #
    #     nn_descent = pynndescent_.make_nn_descent(distances.named_distances['euclidean'], ())
    #     nn = nn_descent(data, n_neighbors=n_neighbors, rng_state=spark.get_rng_state(42), max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)
    #
    #     nn_spark = spark.nn_descent(self.sc, data, n_neighbors=n_neighbors, rng_state=spark.get_rng_state(42), max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)
    #
    #     assert_allclose(nn, nn_spark)
