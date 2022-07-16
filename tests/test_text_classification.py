"""Unit tests for the `text_classification` module."""

import pytest

from src.scandeval.config import BenchmarkConfig
from src.scandeval.dataset_configs import (
    ABSABANK_IMM_CONFIG,
    ANGRY_TWEETS_CONFIG,
    NOREC_CONFIG,
    SCALA_DA_CONFIG,
    SCALA_FO_CONFIG,
    SCALA_IS_CONFIG,
    SCALA_NB_CONFIG,
    SCALA_NN_CONFIG,
    SCALA_SV_CONFIG,
)
from src.scandeval.dataset_tasks import LA, SENT
from src.scandeval.languages import DA, FO, IS, NO, SV
from src.scandeval.text_classification import TextClassificationBenchmark


@pytest.fixture(scope="module")
def benchmark_config():
    yield BenchmarkConfig(
        model_languages=[DA, SV, NO, IS, FO],
        dataset_languages=[DA, SV, NO, IS, FO],
        model_tasks=None,
        dataset_tasks=[LA, SENT],
        raise_error_on_invalid_model=False,
        cache_dir=".scandeval_cache",
        evaluate_train=True,
        use_auth_token=False,
        progress_bar=True,
        save_results=False,
        verbose=True,
        testing=True,
    )


@pytest.fixture(scope="module")
def model_id():
    yield "Maltehb/aelaectra-danish-electra-small-cased"


class TestAngryTweets:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=ANGRY_TWEETS_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestAbsabankImm:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=ABSABANK_IMM_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestNorec:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=NOREC_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestScalaDA:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=SCALA_DA_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestScalaSV:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=SCALA_SV_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestScalaNB:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=SCALA_NB_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestScalaNN:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=SCALA_NN_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestScalaIS:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=SCALA_IS_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestScalaFO:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield TextClassificationBenchmark(
            dataset_config=SCALA_FO_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )
