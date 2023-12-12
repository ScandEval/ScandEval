"""Unit tests for the `question_answering` module."""

from typing import Generator
import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    GERMANQUAD_CONFIG,
    NQII_CONFIG,
    SCANDIQA_DA_CONFIG,
    SCANDIQA_NO_CONFIG,
    SCANDIQA_SV_CONFIG,
)
from scandeval.question_answering import QuestionAnswering


@pytest.fixture(
    scope="module",
    params=[
        SCANDIQA_DA_CONFIG,
        SCANDIQA_NO_CONFIG,
        SCANDIQA_SV_CONFIG,
        NQII_CONFIG,
        GERMANQUAD_CONFIG,
    ],
    ids=[
        "scandiqa-da",
        "scandiqa-no",
        "scandiqa-sv",
        "nqii",
        "germanquad",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield QuestionAnswering(
        dataset_config=request.param,
        benchmark_config=benchmark_config,
    )


def test_encoder_sequence_classification(benchmark_dataset, model_id):
    benchmark_dataset.benchmark(model_id)


def test_decoder_sequence_classification(benchmark_dataset, generative_model_id):
    benchmark_dataset.benchmark(generative_model_id)
