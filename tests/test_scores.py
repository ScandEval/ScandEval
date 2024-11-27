"""Unit tests for the `scores` module."""

from copy import deepcopy
from typing import Generator

import numpy as np
import pytest

from scandeval.scores import aggregate_scores, log_scores
from scandeval.types import ScoreDict


@pytest.fixture(scope="module")
def scores(metric_config) -> Generator[dict[str, list[dict[str, float]]], None, None]:
    """Yield a dictionary of scores."""
    yield dict(
        train=[
            {f"train_{metric_config.name}": 0.70},
            {f"train_{metric_config.name}": 0.75},
            {f"train_{metric_config.name}": 0.80},
        ],
        test=[
            {f"test_{metric_config.name}": 0.50},
            {f"test_{metric_config.name}": 0.55},
            {f"test_{metric_config.name}": 0.60},
        ],
    )


class TestAggregateScores:
    """Unit tests for the `aggregate_scores` function."""

    def test_only_train_scores(self, scores, metric_config):
        """Test that `aggregate_scores` works when only train scores are provided."""
        # Remove the test scores from the scores
        scores_only_train = deepcopy(scores)
        scores_only_train.pop("test")

        # Aggregate scores using the `agg_scores` function
        agg_scores = aggregate_scores(
            scores=scores_only_train, metric_config=metric_config
        )

        # Manually compute the mean and standard error of the scores
        train_scores = [
            dct[f"train_{metric_config.name}"] for dct in scores_only_train["train"]
        ]
        mean = np.mean(train_scores)
        se = 1.96 * np.std(train_scores, ddof=1) / np.sqrt(len(train_scores))

        # Assert that `aggregate_scores` computed the same
        assert agg_scores == dict(train=(mean, se))

    def test_only_test_scores(self, scores, metric_config):
        """Test that `aggregate_scores` works when only test scores are provided."""
        # Remove the train scores from the scores
        scores_only_test = deepcopy(scores)
        scores_only_test.pop("train")

        # Aggregate scores using the `agg_scores` function
        agg_scores = aggregate_scores(
            scores=scores_only_test, metric_config=metric_config
        )

        # Manually compute the mean and standard error of the scores
        test_scores = [
            dct[f"test_{metric_config.name}"] for dct in scores_only_test["test"]
        ]
        mean = np.mean(test_scores)
        se = 1.96 * np.std(test_scores, ddof=1) / np.sqrt(len(test_scores))

        # Assert that `aggregate_scores` computed the same
        assert agg_scores == dict(test=(mean, se))

    def test_all_scores(self, scores, metric_config):
        """Test that `aggregate_scores` works when train/test scores are provided."""
        # Aggregate scores using the `agg_scores` function
        agg_scores = aggregate_scores(scores=scores, metric_config=metric_config)

        # Manually compute the mean and standard error of the train scores
        train_scores = [dct[f"train_{metric_config.name}"] for dct in scores["train"]]
        train_mean = np.mean(train_scores)
        train_se = 1.96 * np.std(train_scores, ddof=1) / np.sqrt(len(train_scores))

        # Manually compute the mean and standard error of the test scores
        test_scores = [dct[f"test_{metric_config.name}"] for dct in scores["test"]]
        test_mean = np.mean(test_scores)
        test_se = 1.96 * np.std(test_scores, ddof=1) / np.sqrt(len(test_scores))

        # Assert that `aggregate_scores` computed the same
        assert agg_scores == dict(
            train=(train_mean, train_se), test=(test_mean, test_se)
        )

    def test_no_scores(self, scores, metric_config):
        """Test that `aggregate_scores` works when no scores are provided."""
        empty_scores = deepcopy(scores)
        empty_scores.pop("train")
        empty_scores.pop("test")
        agg_scores = aggregate_scores(scores=empty_scores, metric_config=metric_config)
        assert agg_scores == dict()


class TestLogScores:
    """Unit tests for the `log_scores` function."""

    @pytest.fixture(scope="class")
    def logged_scores(self, metric_config, scores) -> Generator[ScoreDict, None, None]:
        """Yields the logged scores."""
        yield log_scores(
            dataset_name="dataset",
            metric_configs=[metric_config],
            scores=scores,
            model_id="model_id",
        )

    def test_is_correct_type(self, logged_scores):
        """Test that `log_scores` returns a dictionary."""
        assert isinstance(logged_scores, dict)

    def test_has_correct_keys(self, logged_scores):
        """Test that `log_scores` returns a dictionary with the correct keys."""
        assert sorted(logged_scores.keys()) == ["raw", "total"]

    def test_raw_scores_are_identical_to_input(self, logged_scores, scores):
        """Test that `log_scores` returns the same raw scores as the input."""
        assert logged_scores["raw"] == scores

    def test_total_scores_is_dict(self, logged_scores):
        """Test that `log_scores` returns a dictionary for the total scores."""
        assert isinstance(logged_scores["total"], dict)

    def test_total_scores_keys(self, logged_scores, metric_config):
        """Test that `log_scores` returns a dictionary with the correct keys."""
        assert sorted(logged_scores["total"].keys()) == [
            f"test_{metric_config.name}",
            f"test_{metric_config.name}_se",
            f"train_{metric_config.name}",
            f"train_{metric_config.name}_se",
        ]

    def test_total_scores_values_are_floats(self, logged_scores):
        """Test that `log_scores` returns a dictionary with float values."""
        for val in logged_scores["total"].values():
            assert isinstance(val, float)
