import logging
import numpy as np
import unittest

from numpy.testing import assert_allclose

from pyspark.sql import SparkSession

from pynndescent import distances
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

    def test_build_candidates(self):
        data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        n_vertices = data.shape[0]
        n_neighbors = 2
        max_candidates = 8

        current_graph = spark.init_current_graph(data, n_neighbors, random_state=42)

        new_candidate_neighbors, old_candidate_neighbors =\
            utils.build_candidates(current_graph.copy(), n_vertices, n_neighbors, max_candidates, spark.get_rng_state(42))

        new_candidate_neighbors_spark, old_candidate_neighbors_spark =\
            spark.build_candidates(self.sc, current_graph.copy(), n_vertices, n_neighbors, max_candidates, spark.get_rng_state(42))

        assert_allclose(new_candidate_neighbors_spark, new_candidate_neighbors)
        assert_allclose(old_candidate_neighbors_spark, old_candidate_neighbors)

    def test_nn_descent(self):
        data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        n_neighbors = 2
        max_candidates = 8

        nn_descent = pynndescent_.make_nn_descent(distances.named_distances['euclidean'], ())
        res = nn_descent(data, n_neighbors=n_neighbors, rng_state=spark.get_rng_state(42), max_candidates=max_candidates, n_iters=1, delta=0, rp_tree_init=False)
        print(res)

        # spark here
