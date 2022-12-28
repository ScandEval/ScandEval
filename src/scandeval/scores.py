"""Aggregation of raw scores into the mean and a confidence interval."""

import logging
import warnings
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from .config import MetricConfig

logger = logging.getLogger(__name__)


def log_scores(
    dataset_name: str,
    metric_configs: Sequence[MetricConfig],
    scores: Dict[str, List[Dict[str, float]]],
    model_id: str,
) -> Dict[str, Union[Dict[str, float], Dict[str, List[Dict[str, float]]]]]:
    """Log the scores.

    Args:
        dataset_name (str):
            Name of the dataset.
        metric_configs (sequence of MetricConfig objects):
            Sequence of metrics to log.
        scores (dict):
            The scores that are to be logged. This is a dict with keys 'train' and
            'test', with values being lists of dictionaries full of scores.
        model_id (str):
            The full Hugging Face Hub path to the pretrained transformer model.

    Returns:
        dict:
            A dictionary with keys 'raw_scores' and 'total', with 'raw_scores' being
            identical to `scores` and 'total' being a dictionary with the aggregated
            scores (means and standard errors).
    """
    # Initial logging message
    msg = f"Finished finetuning and evaluation of {model_id} on {dataset_name}."
    logger.info(msg)

    # Initialise the total dict
    total_dict: Dict[str, float] = dict()

    # Logging of the aggregated scores
    for metric_cfg in metric_configs:
        agg_scores = aggregate_scores(scores=scores, metric_config=metric_cfg)
        test_score, test_se = agg_scores["test"]
        test_score, test_score_str = metric_cfg.postprocessing_fn(test_score)
        test_se, test_se_str = metric_cfg.postprocessing_fn(test_se)
        msg = f"{metric_cfg.pretty_name}:\n  - Test: {test_score_str} ± {test_se_str}"

        if "train" in agg_scores.keys():
            train_score, train_se = agg_scores["train"]
            train_score, train_score_str = metric_cfg.postprocessing_fn(train_score)
            train_se, train_se_str = metric_cfg.postprocessing_fn(train_se)
            msg += f"\n  - Train: {train_score_str} ± {train_se_str}"

            # Store the aggregated train scores
            total_dict[f"train_{metric_cfg.name}"] = train_score
            total_dict[f"train_{metric_cfg.name}_se"] = train_se

        # Store the aggregated test scores
        total_dict[f"test_{metric_cfg.name}"] = test_score
        total_dict[f"test_{metric_cfg.name}_se"] = test_se

        # Log the scores
        logger.info(msg)

    # Define a dict with both the raw scores and the aggregated scores
    all_scores: Dict[str, Union[Dict[str, float], Dict[str, List[Dict[str, float]]]]]
    all_scores = dict(raw=scores, total=total_dict)

    # Return the extended scores
    return all_scores


def aggregate_scores(
    scores: Dict[str, List[Dict[str, float]]], metric_config: MetricConfig
) -> Dict[str, Tuple[float, float]]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores (dict):
            Dictionary with the names of the metrics as keys, of the form
            "<split>_<metric_name>", such as "val_f1", and values the metric values.
        metric_config (MetricConfig):
            The configuration of the metric, which is used to collect the correct
            metric from `scores`.

    Returns:
        dict:
            Dictionary with keys among 'train' and 'test', with corresponding values
            being a pair of floats, containing the score and the radius of its 95%
            confidence interval.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results = dict()

        if "train" in scores.keys():
            train_scores = [
                dct[f"train_{metric_config.name}"] for dct in scores["train"]
            ]
            train_score = np.mean(train_scores)

            if len(train_scores) > 1:
                sample_std = np.std(train_scores, ddof=1)
                train_se = sample_std / np.sqrt(len(train_scores))
            else:
                train_se = np.nan

            results["train"] = (train_score, 1.96 * train_se)

        if "test" in scores.keys():
            test_scores = [dct[f"test_{metric_config.name}"] for dct in scores["test"]]
            test_score = np.mean(test_scores)

            if len(test_scores) > 1:
                sample_std = np.std(test_scores, ddof=1)
                test_se = sample_std / np.sqrt(len(test_scores))
            else:
                test_se = np.nan

            results["test"] = (test_score, 1.96 * test_se)

        return results
