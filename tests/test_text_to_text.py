"""Unit tests for the `text_to_text` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
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
    ],
    ids=[
        "nordjylland-news",
        "swedn",
        "no-sammendrag",
        "rrn",
        "mlsum",
        "wiki-lingua-nl",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield TextToText(
        dataset_config=request.param,
        benchmark_config=benchmark_config,
    )


def test_encoder_sequence_classification(benchmark_dataset, model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(model_id)


def test_decoder_sequence_classification(benchmark_dataset, generative_model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(generative_model_id)
