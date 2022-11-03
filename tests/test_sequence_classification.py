"""Unit tests for the `text_classification` module."""

import pytest

from scandeval.config import BenchmarkConfig
from scandeval.dataset_configs import (
    ABSABANK_IMM_CONFIG,
    ANGRY_TWEETS_CONFIG,
    NOREC_CONFIG,
    SCALA_DA_CONFIG,
    SCALA_NB_CONFIG,
    SCALA_NN_CONFIG,
    SCALA_SV_CONFIG,
)
from scandeval.dataset_tasks import LA, SENT
from scandeval.languages import DA, NO, SV
from scandeval.sequence_classification import SequenceClassification


@pytest.fixture(scope="module")
def benchmark_config():
    yield BenchmarkConfig(
        model_languages=[DA, SV, NO],
        dataset_languages=[DA, SV, NO],
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
        (ANGRY_TWEETS_CONFIG, (0.00, 13.79)),
        (ABSABANK_IMM_CONFIG, (0.00, 10.15)),
        (NOREC_CONFIG, (0.00, 16.79)),
        (SCALA_DA_CONFIG, (0.00, 35.31)),
        (SCALA_SV_CONFIG, (0.00, 29.48)),
        (SCALA_NB_CONFIG, (0.00, 32.06)),
        (SCALA_NN_CONFIG, (0.00, 33.91)),
    ],
    ids=[
        "angry-tweets",
        "absabank-imm",
        "norec",
        "scala-da",
        "scala-sv",
        "scala-nb",
        "scala-nn",
    ],
    scope="class",
)
class TestTextClassificationScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, model_id, dataset):
        benchmark = SequenceClassification(
            dataset_config=dataset,
            benchmark_config=benchmark_config,
        )
        yield benchmark.benchmark(model_id)[0]["total"]

    def test_mcc_is_correct(self, scores, correct_scores):
        min_score = scores["test_mcc"] - scores["test_mcc_se"]
        max_score = scores["test_mcc"] + scores["test_mcc_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_macro_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_macro_f1"] - scores["test_macro_f1_se"]
        max_score = scores["test_macro_f1"] + scores["test_macro_f1_se"]
        assert min_score <= correct_scores[1] <= max_score
