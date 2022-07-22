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
        (ANGRY_TWEETS_CONFIG, (-8.5, 18.2)),
        (ABSABANK_IMM_CONFIG, (0.0, 12.9)),
        (NOREC_CONFIG, (1.5, 23.7)),
        (SCALA_DA_CONFIG, (8.5, 36.7)),
        (SCALA_SV_CONFIG, (0.0, 33.0)),
        (SCALA_NB_CONFIG, (-7.2, 32.5)),
        (SCALA_NN_CONFIG, (-11.0, 34.8)),
        (SCALA_IS_CONFIG, (0.0, 34.4)),
        (SCALA_FO_CONFIG, (0.0, 31.7)),
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

    def test_mcc_is_correct(self, scores, correct_scores):
        min_score = scores["test_mcc"] - scores["test_mcc_se"]
        max_score = scores["test_mcc"] + scores["test_mcc_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_macro_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_macro_f1"] - scores["test_macro_f1_se"]
        max_score = scores["test_macro_f1"] + scores["test_macro_f1_se"]
        assert min_score <= correct_scores[1] <= max_score
