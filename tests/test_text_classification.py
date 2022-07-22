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
        (ANGRY_TWEETS_CONFIG, (-8.5, 18.2, 16.6, 5.8)),
        (ABSABANK_IMM_CONFIG, (0.0, 12.9, 0.0, 4.6)),
        (NOREC_CONFIG, (1.5, 23.7, 3.0, 8.7)),
        (SCALA_DA_CONFIG, (8.5, 36.7, 16.6, 6.0)),
        (SCALA_SV_CONFIG, (0.0, 33.0, 0.0, 7.0)),
        (SCALA_NB_CONFIG, (-7.2, 32.5, 14.1, 4.0)),
        (SCALA_NN_CONFIG, (-11.0, 34.8, 21.5, 3.6)),
        (SCALA_IS_CONFIG, (0.0, 34.4, 0.0, 4.9)),
        (SCALA_FO_CONFIG, (0.0, 31.7, 0.0, 1.8)),
    ],
    ids=[
        "absabank-imm",
        "angry-tweets",
        "norec",
        "scala-da",
        "scala-sv",
        "scala-nb",
        "scala-nn",
        "scala-is",
        "scala-fo",
    ],
    scope="class",
)
class TestTextClassificationScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, model_id, dataset):
        benchmark = TextClassificationBenchmark(
            dataset_config=dataset,
            benchmark_config=benchmark_config,
        )
        yield benchmark.benchmark(model_id)["total"]

    def test_mean_mcc_is_correct(self, scores, correct_scores):
        min_score = correct_scores[0] * 0.9
        max_score = correct_scores[0] * 1.1
        assert min_score <= round(scores["test_mcc"], 1) <= max_score

    def test_mean_macro_f1_is_correct(self, scores, correct_scores):
        min_score = correct_scores[1] * 0.9
        max_score = correct_scores[1] * 1.1
        assert min_score <= round(scores["test_macro_f1"], 1) <= max_score

    def test_se_mcc_is_correct(self, scores, correct_scores):
        min_score = correct_scores[2] * 0.9
        max_score = correct_scores[2] * 1.1
        assert min_score <= round(scores["test_mcc_se"], 1) <= max_score

    def test_se_macro_f1_is_correct(self, scores, correct_scores):
        min_score = correct_scores[3] * 0.9
        max_score = correct_scores[3] * 1.1
        assert min_score <= round(scores["test_macro_f1_se"], 1) <= max_score
