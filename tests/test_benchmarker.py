"""Unit tests for the `benchmarker` module."""

from typing import TypedDict

import pytest
from scandeval.benchmarker import BenchmarkResult, model_has_been_benchmarked
from scandeval.types import SCORE_DICT


class DataKwargs(TypedDict):
    """Helper dict with keyword arguments for `BenchmarkResult`, to avoid redundancy."""

    num_model_parameters: int
    max_sequence_length: int
    vocabulary_size: int
    dataset_languages: list[str]
    task: str
    results: SCORE_DICT


DATA_KWARGS = DataKwargs(
    num_model_parameters=100,
    max_sequence_length=100,
    vocabulary_size=100,
    dataset_languages=["da"],
    task="task",
    results=dict(),
)


@pytest.mark.parametrize(
    argnames=[
        "model_id",
        "dataset",
        "few_shot",
        "validation_split",
        "benchmark_results",
        "expected",
    ],
    argvalues=[
        (
            "model",
            "dataset",
            False,
            False,
            [],
            False,
        ),
        (
            "model",
            "dataset",
            False,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                )
            ],
            True,
        ),
        (
            "model",
            "dataset",
            False,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="another-dataset",
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                )
            ],
            False,
        ),
        (
            "model",
            "dataset",
            True,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                )
            ],
            False,
        ),
        (
            "model",
            "dataset",
            True,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=True,
                    validation_split=False,
                    **DATA_KWARGS,
                )
            ],
            True,
        ),
        (
            "model",
            "dataset",
            False,
            True,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                )
            ],
            False,
        ),
        (
            "model",
            "dataset",
            False,
            True,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=False,
                    validation_split=True,
                    **DATA_KWARGS,
                )
            ],
            True,
        ),
        (
            "model",
            "dataset",
            False,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                ),
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                ),
            ],
            True,
        ),
    ],
    ids=[
        "empty benchmark results",
        "model has been benchmarked",
        "model has not been benchmarked",
        "model few-shot has not been benchmarked",
        "model few-shot has been benchmarked",
        "model validation split has not been benchmarked",
        "model validation split has been benchmarked",
        "model has been benchmarked twice",
    ],
)
def test_model_has_been_benchmarked(
    model_id: str,
    dataset: str,
    few_shot: bool,
    validation_split: bool,
    benchmark_results: list[BenchmarkResult],
    expected: bool,
) -> None:
    """Test whether we can correctly check if a model has been benchmarked."""
    benchmarked = model_has_been_benchmarked(
        model_id=model_id,
        dataset=dataset,
        few_shot=few_shot,
        validation_split=validation_split,
        benchmark_results=benchmark_results,
    )
    assert benchmarked == expected
