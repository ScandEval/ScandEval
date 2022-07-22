"""Unit tests for the `ner` module."""

import warnings

import pytest
from sklearn.exceptions import UndefinedMetricWarning

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
        evaluate_train=False,
        use_auth_token=False,
        progress_bar=False,
        save_results=False,
        verbose=False,
        testing=True,
    )


@pytest.fixture(scope="module")
def model_id():
    yield "Maltehb/aelaectra-danish-electra-small-cased"


@pytest.mark.parametrize(
    argnames=["dataset", "correct_scores"],
    argvalues=[
        (DANE_CONFIG, (1.6, 0.7)),
        (SUC3_CONFIG, (0.3, 0.6)),
        (NORNE_NB_CONFIG, (0.8, 0.8)),
        (NORNE_NN_CONFIG, (0.6, 0.5)),
        (MIM_GOLD_NER_CONFIG, (0.9, 1.0)),
        (WIKIANN_FO_CONFIG, (0.8, 0.8)),
    ],
    ids=[
        "dane",
        "suc3",
        "norne_nb",
        "norne_nn",
        "mim_gold_ner",
        "wikiann_fo",
    ],
    scope="class",
)
class TestNerScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, model_id, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            benchmark = NERBenchmark(
                dataset_config=dataset,
                benchmark_config=benchmark_config,
            )
            yield benchmark.benchmark(model_id)["total"]

    def test_micro_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_micro_f1"] - scores["test_micro_f1_se"]
        max_score = scores["test_micro_f1"] + scores["test_micro_f1_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_micro_f1_no_misc_is_correct(self, scores, correct_scores):
        min_score = scores["test_micro_f1_no_misc"] - scores["test_micro_f1_no_misc_se"]
        max_score = scores["test_micro_f1_no_misc"] + scores["test_micro_f1_no_misc_se"]
        assert min_score <= correct_scores[1] <= max_score
