"""Unit tests for the `sequence_classification` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    ANGRY_TWEETS_CONFIG,
    ARC_DA_CONFIG,
    ARC_DE_CONFIG,
    ARC_NL_CONFIG,
    ARC_SV_CONFIG,
    DUTCH_SOCIAL_CONFIG,
    HELLASWAG_DA_CONFIG,
    HELLASWAG_DE_CONFIG,
    HELLASWAG_NL_CONFIG,
    HELLASWAG_SV_CONFIG,
    MMLU_DA_CONFIG,
    MMLU_DE_CONFIG,
    MMLU_NL_CONFIG,
    MMLU_SV_CONFIG,
    NOREC_CONFIG,
    SB10K_CONFIG,
    SCALA_DA_CONFIG,
    SCALA_DE_CONFIG,
    SCALA_EN_CONFIG,
    SCALA_FO_CONFIG,
    SCALA_IS_CONFIG,
    SCALA_NB_CONFIG,
    SCALA_NL_CONFIG,
    SCALA_NN_CONFIG,
    SCALA_SV_CONFIG,
    SST5_CONFIG,
    SWEREC_CONFIG,
)
from scandeval.sequence_classification import SequenceClassification


@pytest.fixture(
    scope="module",
    params=[
        ANGRY_TWEETS_CONFIG,
        SWEREC_CONFIG,
        NOREC_CONFIG,
        SB10K_CONFIG,
        DUTCH_SOCIAL_CONFIG,
        SST5_CONFIG,
        SCALA_DA_CONFIG,
        SCALA_SV_CONFIG,
        SCALA_NB_CONFIG,
        SCALA_NN_CONFIG,
        SCALA_IS_CONFIG,
        SCALA_FO_CONFIG,
        SCALA_DE_CONFIG,
        SCALA_NL_CONFIG,
        SCALA_EN_CONFIG,
        MMLU_DA_CONFIG,
        MMLU_SV_CONFIG,
        MMLU_DE_CONFIG,
        MMLU_NL_CONFIG,
        HELLASWAG_DA_CONFIG,
        HELLASWAG_SV_CONFIG,
        HELLASWAG_DE_CONFIG,
        HELLASWAG_NL_CONFIG,
        ARC_DA_CONFIG,
        ARC_SV_CONFIG,
        ARC_DE_CONFIG,
        ARC_NL_CONFIG,
    ],
    ids=[
        "angry-tweets",
        "swerec",
        "norec",
        "sb10k",
        "dutch-social",
        "sst5",
        "scala-da",
        "scala-sv",
        "scala-nb",
        "scala-nn",
        "scala-is",
        "scala-fo",
        "scala-de",
        "scala-nl",
        "scala-en",
        "mmlu-da",
        "mmlu-sv",
        "mmlu-de",
        "mmlu-nl",
        "hellaswag-da",
        "hellaswag-sv",
        "hellaswag-de",
        "hellaswag-nl",
        "arc-da",
        "arc-sv",
        "arc-de",
        "arc-nl",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield SequenceClassification(
        dataset_config=request.param,
        benchmark_config=benchmark_config,
    )


def test_encoder_benchmarking(benchmark_dataset, model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(model_id)


def test_decoder_sequence_benchmarking(benchmark_dataset, generative_model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(generative_model_id)
