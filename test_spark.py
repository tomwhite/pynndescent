import logging
import unittest

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
            SparkSession.builder.master("local[2]")
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

    def test_1(self):
        X = utils.make_heap(6, 2)
        s = X.shape
        rdd = spark.to_rdd(self.sc, X, (s[0], 4, s[2])) # chunk on second axis
        print(rdd.collect())
