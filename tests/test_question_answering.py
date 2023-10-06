"""Unit tests for the `question_answering` module."""

import pytest

from scandeval.dataset_configs import (
    SCANDIQA_DA_CONFIG,
    SCANDIQA_NO_CONFIG,
    SCANDIQA_SV_CONFIG,
)
from scandeval.question_answering import QuestionAnswering


@pytest.mark.parametrize(
    argnames=["dataset", "correct_scores"],
    argvalues=[
        (SCANDIQA_DA_CONFIG, (0.00, 4.18)),
        (SCANDIQA_NO_CONFIG, (0.00, 4.15)),
        (SCANDIQA_SV_CONFIG, (0.00, 4.44)),
    ],
    ids=[
        "scandiqa-da",
        "scandiqa-no",
        "scandiqa-sv",
    ],
    scope="class",
)
class TestScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, model_id, dataset):
        benchmark = QuestionAnswering(
            dataset_config=dataset,
            benchmark_config=benchmark_config,
        )
        yield benchmark.benchmark(model_id)[0]["total"]

    def test_em_is_correct(self, scores, correct_scores):
        min_score = scores["test_em"] - scores["test_em_se"]
        max_score = scores["test_em"] + scores["test_em_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_f1"] - scores["test_f1_se"]
        max_score = scores["test_f1"] + scores["test_f1_se"]
        assert min_score <= correct_scores[1] <= max_score
