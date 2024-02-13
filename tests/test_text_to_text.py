"""Unit tests for the `text_to_text` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    CNN_DAILYMAIL_CONFIG,
    MLSUM_CONFIG,
    NO_SAMMENDRAG_CONFIG,
    NORDJYLLAND_NEWS_CONFIG,
    RRN_CONFIG,
    SWEDN_CONFIG,
    WIKI_LINGUA_NL_CONFIG,
)
from scandeval.text_to_text import TextToText


@pytest.fixture(
    scope="module",
    params=[
        NORDJYLLAND_NEWS_CONFIG,
        SWEDN_CONFIG,
        NO_SAMMENDRAG_CONFIG,
        RRN_CONFIG,
        MLSUM_CONFIG,
        WIKI_LINGUA_NL_CONFIG,
        CNN_DAILYMAIL_CONFIG,
    ],
    ids=[
        "nordjylland-news",
        "swedn",
        "no-sammendrag",
        "rrn",
        "mlsum",
        "wiki-lingua-nl",
        "cnn-dailymail",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    """Yields a text-to-text benchmark dataset."""
    yield TextToText(dataset_config=request.param, benchmark_config=benchmark_config)


def test_decoder_benchmarking(benchmark_dataset, generative_model_id):
    """Test that decoder models can be benchmarked on text-to-text tasks."""
    with does_not_raise():
        benchmark_dataset.benchmark(generative_model_id)
