import logging
import numpy as np
import unittest

from numpy.testing import assert_allclose

from pyspark.sql import SparkSession

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

    def setUp(self):
        pass

    # def test_1(self):
    #     X = utils.make_heap(6, 2)
    #     s = X.shape
    #     rdd = spark.to_rdd(self.sc, X, (s[0], 4, s[2])) # chunk on second axis
    #     print(rdd.collect())
    #
    # def test_init_current_graph(self):
    #     data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #     current_graph = spark.init_current_graph(data, 2, random_state=42)
    #     print(current_graph)

    def test_build_candidates(self):
        data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        current_graph = spark.init_current_graph(data, 2, random_state=42)
        new_candidate_neighbors, old_candidate_neighbors = utils.build_candidates(current_graph, current_graph.shape[1], 2, 8, spark.get_rng_state(42))

        # print(new_candidate_neighbors)
        # print(old_candidate_neighbors)

        print("spark!")
        new_candidate_neighbors_spark = spark.build_candidates(self.sc, current_graph, current_graph.shape[1], 2, 8, spark.get_rng_state(42))

        assert_allclose(new_candidate_neighbors_spark, new_candidate_neighbors)