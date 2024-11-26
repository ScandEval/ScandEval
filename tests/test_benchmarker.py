"""Unit tests for the `benchmarker` module."""

import json
from pathlib import Path
from typing import Generator, TypedDict

import pytest

from scandeval import __version__
from scandeval.benchmarker import (
    Benchmarker,
    BenchmarkResult,
    model_has_been_benchmarked,
)
from scandeval.types import ScoreDict


class DataKwargs(TypedDict):
    """Helper dict with keyword arguments for `BenchmarkResult`, to avoid redundancy."""

    num_model_parameters: int
    max_sequence_length: int
    vocabulary_size: int
    dataset_languages: list[str]
    task: str
    results: ScoreDict


DATA_KWARGS = DataKwargs(
    num_model_parameters=100,
    max_sequence_length=100,
    vocabulary_size=100,
    dataset_languages=["da"],
    task="task",
    results=dict(),
)


def test_benchmarker_initialisation():
    """Test that the `Benchmarker` class can be initialised."""
    Benchmarker()


class TestBenchmarkResult:
    """Tests related to the `BenchmarkResult` class."""

    @pytest.fixture(scope="class")
    def benchmark_result(self) -> Generator[BenchmarkResult, None, None]:
        """Fixture for a `BenchmarkResult` object."""
        yield BenchmarkResult(
            dataset="dataset",
            model="model",
            generative=False,
            few_shot=True,
            validation_split=False,
            **DATA_KWARGS,
        )

    @pytest.fixture(scope="class")
    def results_path(self) -> Generator[Path, None, None]:
        """Fixture for a `Path` object to a results file."""
        results_path = Path(".scandeval_cache/test_results.jsonl")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        yield results_path

    def test_benchmark_result_parameters(self, benchmark_result):
        """Test that the `BenchmarkResult` parameters are correct."""
        assert benchmark_result.dataset == "dataset"
        assert benchmark_result.model == "model"
        assert benchmark_result.generative is False
        assert benchmark_result.few_shot is True
        assert benchmark_result.validation_split is False
        assert benchmark_result.num_model_parameters == 100
        assert benchmark_result.max_sequence_length == 100
        assert benchmark_result.vocabulary_size == 100
        assert benchmark_result.dataset_languages == ["da"]
        assert benchmark_result.task == "task"
        assert benchmark_result.results == dict()
        assert benchmark_result.scandeval_version == __version__

    @pytest.mark.parametrize(
        argnames=["config", "expected"],
        argvalues=[
            (
                dict(
                    dataset="dataset",
                    model="model",
                    few_shot=True,
                    validation_split=False,
                    **DATA_KWARGS,
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    generative=False,
                    few_shot=True,
                    validation_split=False,
                    **DATA_KWARGS,
                ),
            ),
            (
                dict(
                    dataset="dataset",
                    model="model (few-shot)",
                    validation_split=False,
                    **DATA_KWARGS,
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    generative=True,
                    few_shot=True,
                    validation_split=False,
                    **DATA_KWARGS,
                ),
            ),
            (
                dict(
                    dataset="dataset", model="model (val)", few_shot=True, **DATA_KWARGS
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    generative=False,
                    few_shot=True,
                    validation_split=True,
                    **DATA_KWARGS,
                ),
            ),
            (
                dict(dataset="dataset", model="model (few-shot, val)", **DATA_KWARGS),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    generative=True,
                    few_shot=True,
                    validation_split=True,
                    **DATA_KWARGS,
                ),
            ),
        ],
        ids=[
            "normal case",
            "few-shot model name",
            "validation split model name",
            "few-shot and validation split model name",
        ],
    )
    def test_from_dict(self, config, expected):
        """Test that `BenchmarkResult.from_dict` works as expected."""
        assert BenchmarkResult.from_dict(config) == expected

    def test_append_to_results(self, benchmark_result, results_path):
        """Test that `BenchmarkResult.append_to_results` works as expected."""
        results_path.unlink(missing_ok=True)
        results_path.touch(exist_ok=True)

        benchmark_result.append_to_results(results_path=results_path)
        json_str = json.dumps(
            dict(
                dataset=benchmark_result.dataset,
                task=benchmark_result.task,
                dataset_languages=benchmark_result.dataset_languages,
                model=benchmark_result.model,
                results=benchmark_result.results,
                num_model_parameters=benchmark_result.num_model_parameters,
                max_sequence_length=benchmark_result.max_sequence_length,
                vocabulary_size=benchmark_result.vocabulary_size,
                generative=benchmark_result.generative,
                few_shot=benchmark_result.few_shot,
                validation_split=benchmark_result.validation_split,
                scandeval_version=benchmark_result.scandeval_version,
            )
        )
        assert results_path.read_text() == f"\n{json_str}"

        benchmark_result.append_to_results(results_path=results_path)
        assert results_path.read_text() == f"\n{json_str}\n{json_str}"

        results_path.unlink(missing_ok=True)


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
        ("model", "dataset", False, False, [], False),
        (
            "model",
            "dataset",
            False,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    generative=False,
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
                    generative=False,
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
                    generative=True,
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
                    generative=True,
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
            True,
            False,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    generative=False,
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
            True,
            [
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    generative=False,
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
                    generative=False,
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
                    generative=False,
                    few_shot=False,
                    validation_split=False,
                    **DATA_KWARGS,
                ),
                BenchmarkResult(
                    model="model",
                    dataset="dataset",
                    generative=False,
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
        "model few-shot has been benchmarked, but not generative",
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
