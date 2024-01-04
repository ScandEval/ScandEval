"""Unit tests for the `named_entity_recognition` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    CONLL_EN_CONFIG,
    CONLL_NL_CONFIG,
    DANSK_CONFIG,
    FONE_CONFIG,
    GERMEVAL_CONFIG,
    MIM_GOLD_NER_CONFIG,
    NORNE_NB_CONFIG,
    NORNE_NN_CONFIG,
    SUC3_CONFIG,
)
from scandeval.exceptions import InvalidBenchmark
from scandeval.named_entity_recognition import NamedEntityRecognition
from scandeval.utils import GENERATIVE_DATASET_TASKS


@pytest.fixture(
    scope="module",
    params=[
        DANSK_CONFIG,
        SUC3_CONFIG,
        NORNE_NB_CONFIG,
        NORNE_NN_CONFIG,
        MIM_GOLD_NER_CONFIG,
        FONE_CONFIG,
        GERMEVAL_CONFIG,
        CONLL_NL_CONFIG,
        CONLL_EN_CONFIG,
    ],
    ids=[
        "dansk",
        "suc3",
        "norne_nb",
        "norne_nn",
        "mim-gold-ner",
        "fone",
        "germeval",
        "conll-nl",
        "conll-en",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield NamedEntityRecognition(
        dataset_config=request.param,
        benchmark_config=benchmark_config,
    )


def test_encoder_benchmarking(benchmark_dataset, model_id):
    if benchmark_dataset.dataset_config.task.name in GENERATIVE_DATASET_TASKS:
        with pytest.raises(InvalidBenchmark):
            benchmark_dataset.benchmark(model_id)
    else:
        with does_not_raise():
            benchmark_dataset.benchmark(model_id)


def test_decoder_benchmarking(benchmark_dataset, generative_model_id):
    with does_not_raise():
        benchmark_dataset.benchmark(generative_model_id)
