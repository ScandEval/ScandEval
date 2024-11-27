"""Unit tests for the `sequence_classification` module."""

import os
from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest

from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.exceptions import InvalidBenchmark
from scandeval.languages import DA
from scandeval.sequence_classification import SequenceClassification
from scandeval.tasks import COMMON_SENSE, KNOW, SENT
from scandeval.utils import GENERATIVE_DATASET_TASKS


@pytest.fixture(
    scope="module",
    params=[
        dataset_config
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config.task in [SENT, KNOW, COMMON_SENSE]
        and (
            os.getenv("TEST_ALL_DATASETS", "0") == "1"
            or (not dataset_config.unofficial and dataset_config.languages == [DA])
        )
    ],
    ids=lambda dataset_config: dataset_config.name,
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    """Yields a sequence classification benchmark dataset."""
    yield SequenceClassification(
        dataset_config=request.param, benchmark_config=benchmark_config
    )


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
def test_encoder_benchmarking(benchmark_dataset, model_id):
    """Test that the encoder can be benchmarked on sequence classification datasets."""
    if benchmark_dataset.dataset_config.task.name in GENERATIVE_DATASET_TASKS:
        with pytest.raises(InvalidBenchmark):
            benchmark_dataset.benchmark(model_id)
    else:
        with does_not_raise():
            benchmark_dataset.benchmark(model_id)


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
def test_decoder_sequence_benchmarking(
    benchmark_dataset, generative_model_id, generative_model_and_tokenizer
):
    """Test that the decoder can be benchmarked on sequence classification datasets."""
    model, tokenizer = generative_model_and_tokenizer
    with does_not_raise():
        benchmark_dataset.benchmark(
            model_id=generative_model_id, model=model, tokenizer=tokenizer
        )
