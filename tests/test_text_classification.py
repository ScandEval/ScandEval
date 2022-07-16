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
        (ANGRY_TWEETS_CONFIG, (-8.46, 18.21, 16.58, 5.78)),
        (ABSABANK_IMM_CONFIG, (0.00, 12.89, 0.00, 4.64)),
        (NOREC_CONFIG, (1.52, 23.70, 2.98, 8.70)),
        (SCALA_DA_CONFIG, (8.46, 36.72, 16.58, 5.96)),
        (SCALA_SV_CONFIG, (0.00, 32.96, 0.00, 7.20)),
        (SCALA_NB_CONFIG, (-7.21, 32.47, 14.13, 3.99)),
        (SCALA_NN_CONFIG, (-10.95, 34.81, 21.46, 3.58)),
        (SCALA_IS_CONFIG, (0.00, 34.43, 0.00, 4.93)),
        (SCALA_FO_CONFIG, (0.00, 31.72, 0.00, 1.78)),
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
        assert round(scores["test_mcc"], 2) == correct_scores[0]

    def test_mean_macro_f1_is_correct(self, scores, correct_scores):
        assert round(scores["test_macro_f1"], 2) == correct_scores[1]

    def test_se_mcc_is_correct(self, scores, correct_scores):
        assert round(scores["test_mcc_se"], 2) == correct_scores[2]

    def test_se_macro_f1_is_correct(self, scores, correct_scores):
        assert round(scores["test_macro_f1_se"], 2) == correct_scores[3]
