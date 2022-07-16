"""Unit tests for the `ner` module."""

import pytest

from src.scandeval.config import BenchmarkConfig
from src.scandeval.dataset_configs import (
    DANE_CONFIG,
    MIM_GOLD_NER_CONFIG,
    NORNE_NB_CONFIG,
    NORNE_NN_CONFIG,
    SUC3_CONFIG,
    WIKIANN_FO_CONFIG,
)
from src.scandeval.dataset_tasks import NER
from src.scandeval.languages import DA, FO, IS, NO, SV
from src.scandeval.ner import NERBenchmark


@pytest.fixture(scope="module")
def benchmark_config():
    yield BenchmarkConfig(
        model_languages=[DA, SV, NO, IS, FO],
        dataset_languages=[DA, SV, NO, IS, FO],
        model_tasks=None,
        dataset_tasks=[NER],
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


class TestDane:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield NERBenchmark(
            dataset_config=DANE_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestSuc3:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield NERBenchmark(
            dataset_config=SUC3_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestNorneNB:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield NERBenchmark(
            dataset_config=NORNE_NB_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestNorneNN:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield NERBenchmark(
            dataset_config=NORNE_NN_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestMimGoldNer:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield NERBenchmark(
            dataset_config=MIM_GOLD_NER_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )


class TestWikiAnnFo:
    @pytest.fixture(scope="class")
    def benchmark(self, benchmark_config):
        yield NERBenchmark(
            dataset_config=WIKIANN_FO_CONFIG,
            benchmark_config=benchmark_config,
        )

    def test_scores_are_correct(self, benchmark, model_id):
        assert benchmark.benchmark(model_id) == dict(
            test=[{"test_micro_f1": 0.5, "test_micro_f1_no_misc": 0.5}]
        )
