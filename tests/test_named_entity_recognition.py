"""Unit tests for the `ner` module."""

import warnings

import pytest
from sklearn.exceptions import UndefinedMetricWarning

from scandeval.dataset_configs import (
    DANE_CONFIG,
    NORNE_NB_CONFIG,
    NORNE_NN_CONFIG,
    SUC3_CONFIG,
)
from scandeval.named_entity_recognition import NamedEntityRecognition


@pytest.mark.parametrize(
    argnames=["dataset", "correct_scores"],
    argvalues=[
        (DANE_CONFIG, (1.87, 1.22)),
        (SUC3_CONFIG, (1.86, 2.26)),
        (NORNE_NB_CONFIG, (1.83, 1.94)),
        (NORNE_NN_CONFIG, (1.14, 1.20)),
    ],
    ids=[
        "dane",
        "suc3",
        "norne_nb",
        "norne_nn",
    ],
    scope="class",
)
class TestScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, model_id, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            benchmark = NamedEntityRecognition(
                dataset_config=dataset,
                benchmark_config=benchmark_config,
            )
            yield benchmark.benchmark(model_id)[0]["total"]

    def test_micro_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_micro_f1"] - scores["test_micro_f1_se"]
        max_score = scores["test_micro_f1"] + scores["test_micro_f1_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_micro_f1_no_misc_is_correct(self, scores, correct_scores):
        min_score = scores["test_micro_f1_no_misc"] - scores["test_micro_f1_no_misc_se"]
        max_score = scores["test_micro_f1_no_misc"] + scores["test_micro_f1_no_misc_se"]
        assert min_score <= correct_scores[1] <= max_score
