"""Unit tests for the `sequence_classification` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    ANGRY_TWEETS_CONFIG,
    DANSKE_TALEMAADER_CONFIG,
    DUTCH_SOCIAL_CONFIG,
    HELLASWAG_CONFIG,
    HELLASWAG_DA_CONFIG,
    HELLASWAG_DE_CONFIG,
    HELLASWAG_IS_CONFIG,
    HELLASWAG_NL_CONFIG,
    HELLASWAG_NO_CONFIG,
    HELLASWAG_SV_CONFIG,
    MMLU_CONFIG,
    MMLU_DA_CONFIG,
    MMLU_DE_CONFIG,
    MMLU_IS_CONFIG,
    MMLU_NL_CONFIG,
    MMLU_NO_CONFIG,
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
from scandeval.exceptions import InvalidBenchmark
from scandeval.sequence_classification import SequenceClassification
from scandeval.utils import GENERATIVE_DATASET_TASKS


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
        DANSKE_TALEMAADER_CONFIG,
        MMLU_DA_CONFIG,
        MMLU_SV_CONFIG,
        MMLU_NO_CONFIG,
        MMLU_IS_CONFIG,
        MMLU_DE_CONFIG,
        MMLU_NL_CONFIG,
        MMLU_CONFIG,
        HELLASWAG_DA_CONFIG,
        HELLASWAG_SV_CONFIG,
        HELLASWAG_NO_CONFIG,
        HELLASWAG_IS_CONFIG,
        HELLASWAG_DE_CONFIG,
        HELLASWAG_NL_CONFIG,
        HELLASWAG_CONFIG,
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
        "danske-talemaader",
        "mmlu-da",
        "mmlu-sv",
        "mmlu-no",
        "mmlu-is",
        "mmlu-de",
        "mmlu-nl",
        "mmlu",
        "hellaswag-da",
        "hellaswag-sv",
        "hellaswag-no",
        "hellaswag-is",
        "hellaswag-de",
        "hellaswag-nl",
        "hellaswag",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    """Yields a sequence classification benchmark dataset."""
    yield SequenceClassification(
        dataset_config=request.param, benchmark_config=benchmark_config
    )


def test_encoder_benchmarking(benchmark_dataset, model_id):
    """Test that the encoder can be benchmarked on sequence classification datasets."""
    if benchmark_dataset.dataset_config.task.name in GENERATIVE_DATASET_TASKS:
        with pytest.raises(InvalidBenchmark):
            benchmark_dataset.benchmark(model_id)
    else:
        with does_not_raise():
            benchmark_dataset.benchmark(model_id)


def test_decoder_sequence_benchmarking(benchmark_dataset, generative_model_id):
    """Test that the decoder can be benchmarked on sequence classification datasets."""
    with does_not_raise():
        benchmark_dataset.benchmark(generative_model_id)
