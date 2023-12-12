"""Unit tests for the `sequence_classification` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    ANGRY_TWEETS_CONFIG,
    DUTCH_SOCIAL_CONFIG,
    NOREC_CONFIG,
    SB10K_CONFIG,
    SCALA_DA_CONFIG,
    SCALA_DE_CONFIG,
    SCALA_NB_CONFIG,
    SCALA_NL_CONFIG,
    SCALA_NN_CONFIG,
    SCALA_SV_CONFIG,
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
        SCALA_DA_CONFIG,
        SCALA_SV_CONFIG,
        SCALA_NB_CONFIG,
        SCALA_NN_CONFIG,
        SCALA_DE_CONFIG,
        SCALA_NL_CONFIG,
    ],
    ids=[
        "angry-tweets",
        "swerec",
        "norec",
        "sb10k",
        "dutch-social",
        "scala-da",
        "scala-sv",
        "scala-nb",
        "scala-nn",
        "scala-de",
        "scala-nl",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield SequenceClassification(
        dataset_config=request.param,
        benchmark_config=benchmark_config,
    )


def test_encoder_sequence_classification(benchmark_dataset, model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(model_id)


def test_decoder_sequence_classification(benchmark_dataset, generative_model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(generative_model_id)
