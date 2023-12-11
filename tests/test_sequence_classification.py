"""Unit tests for the `sequence_classification` module."""

import pytest

from scandeval.dataset_configs import (
    ANGRY_TWEETS_CONFIG,
    DUTCH_SOCIAL_CONFIG,
    NOREC_CONFIG,
    SB10K_CONFIG,
    SCALA_DA_CONFIG,
    SCALA_DE_CONFIG,
    SCALA_NB_CONFIG,
    SCALA_NL_CONFIG,
    SCALA_NN_CONFIG,
    SCALA_SV_CONFIG,
    SWEREC_CONFIG,
)
from scandeval.sequence_classification import SequenceClassification


@pytest.mark.parametrize(
    argnames=["dataset", "correct_scores"],
    argvalues=[
        (ANGRY_TWEETS_CONFIG, (-4.43, 23.00)),
        (SWEREC_CONFIG, (0.00, 23.07)),
        (NOREC_CONFIG, (2.32, 25.12)),
        (SB10K_CONFIG, (-1000, -1000)),
        (DUTCH_SOCIAL_CONFIG, (-1000, -1000)),
        (SCALA_DA_CONFIG, (3.69, 36.61)),
        (SCALA_SV_CONFIG, (2.85, 36.84)),
        (SCALA_NB_CONFIG, (0.00, 30.71)),
        (SCALA_NN_CONFIG, (0.00, 33.39)),
        (SCALA_DE_CONFIG, (-1000, -1000)),
        (SCALA_NL_CONFIG, (-1000, -1000)),
    ],
    ids=[
        "angry-tweets",
        "swerec",
        "norec",
        "sb10k",
        "dutch-social",
        "scala-da",
        "scala-sv",
        "scala-nb",
        "scala-nn",
        "scala-de",
        "scala-nl",
    ],
    scope="class",
)
class TestScores:
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
