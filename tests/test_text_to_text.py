"""Unit tests for the `text_to_text` module."""

import os
from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest

from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.languages import DA
from scandeval.tasks import SUMM
from scandeval.text_to_text import TextToText


@pytest.fixture(
    scope="module",
    params=[
        dataset_config
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config.task == SUMM
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
    """Yields a text-to-text benchmark dataset."""
    yield TextToText(dataset_config=request.param, benchmark_config=benchmark_config)


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
def test_decoder_benchmarking(
    benchmark_dataset, generative_model_id, generative_model_and_tokenizer
):
    """Test that decoder models can be benchmarked on text-to-text tasks."""
    model, tokenizer = generative_model_and_tokenizer
    with does_not_raise():
        benchmark_dataset.benchmark(
            model_id=generative_model_id, model=model, tokenizer=tokenizer
        )
