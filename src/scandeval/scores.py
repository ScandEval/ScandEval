"""Aggregation of raw scores into the mean and a confidence interval."""

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import MetricConfig
    from .types import ScoreDict

logger = logging.getLogger(__package__)


def log_scores(
    dataset_name: str,
    metric_configs: list["MetricConfig"],
    scores: dict[str, list[dict[str, float]]],
    model_id: str,
) -> "ScoreDict":
    """Log the scores.

    Args:
        dataset_name:
            Name of the dataset.
        metric_configs:
            List of metrics to log.
        scores:
            The scores that are to be logged. This is a dict with keys 'train' and
            'test', with values being lists of dictionaries full of scores.
        model_id:
            The full Hugging Face Hub path to the pretrained transformer model.

    Returns:
        A dictionary with keys 'raw_scores' and 'total', with 'raw_scores' being
        identical to `scores` and 'total' being a dictionary with the aggregated scores
        (means and standard errors).
    """
    logger.info(f"Finished evaluation of {model_id} on {dataset_name}.")

    total_dict: dict[str, float] = dict()

    # Logging of the aggregated scores
    for metric_cfg in metric_configs:
        agg_scores = aggregate_scores(scores=scores, metric_config=metric_cfg)
        test_score, test_se = agg_scores["test"]
        test_score, test_score_str = metric_cfg.postprocessing_fn(test_score)
        test_se, test_se_str = metric_cfg.postprocessing_fn(test_se)
        msg = f"{metric_cfg.pretty_name}:"

        if "train" in agg_scores.keys():
            train_score, train_se = agg_scores["train"]
            train_score, train_score_str = metric_cfg.postprocessing_fn(train_score)
            train_se, train_se_str = metric_cfg.postprocessing_fn(train_se)
            msg += f"\n  - Test: {test_score_str} ± {test_se_str}"
            msg += f"\n  - Train: {train_score_str} ± {train_se_str}"

            # Store the aggregated train scores
            total_dict[f"train_{metric_cfg.name}"] = train_score
            total_dict[f"train_{metric_cfg.name}_se"] = train_se

        else:
            msg += f" {test_score_str} ± {test_se_str}"

        # Store the aggregated test scores
        total_dict[f"test_{metric_cfg.name}"] = test_score
        total_dict[f"test_{metric_cfg.name}_se"] = test_se

        # Log the scores
        logger.info(msg)

    # Define a dict with both the raw scores and the aggregated scores
    all_scores: dict[str, dict[str, float] | dict[str, list[dict[str, float]]]]
    all_scores = dict(raw=scores, total=total_dict)

    # Return the extended scores
    return all_scores


def aggregate_scores(
    scores: dict[str, list[dict[str, float]]], metric_config: "MetricConfig"
) -> dict[str, tuple[float, float]]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores:
            Dictionary with the names of the metrics as keys, of the form
            "<split>_<metric_name>", such as "val_f1", and values the metric values.
        metric_config:
            The configuration of the metric, which is used to collect the correct
            metric from `scores`.

    Returns:
        Dictionary with keys among 'train' and 'test', with corresponding values being
        a pair of floats, containing the score and the radius of its 95% confidence
        interval.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results = dict()

        if "train" in scores.keys():
            train_scores = [
                (
                    dct[metric_config.name]
                    if metric_config.name in dct
                    else dct[f"train_{metric_config.name}"]
                )
                for dct in scores["train"]
            ]
            train_score = np.mean(train_scores)

            if len(train_scores) > 1:
                sample_std = np.std(train_scores, ddof=1)
                train_se = sample_std / np.sqrt(len(train_scores))
            else:
                train_se = np.nan

            results["train"] = (train_score, 1.96 * train_se)

        if "test" in scores.keys():
            test_scores = [
                (
                    dct[metric_config.name]
                    if metric_config.name in dct
                    else dct[f"test_{metric_config.name}"]
                )
                for dct in scores["test"]
            ]
            test_score = np.mean(test_scores)

            if len(test_scores) > 1:
                sample_std = np.std(test_scores, ddof=1)
                test_se = sample_std / np.sqrt(len(test_scores))
            else:
                test_se = np.nan

            results["test"] = (test_score, 1.96 * test_se)

        return results
