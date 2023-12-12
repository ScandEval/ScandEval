"""Unit tests for the `named_entity_recognition` module."""

from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset

from scandeval.dataset_configs import (
    CONLL_NL_CONFIG,
    DANE_CONFIG,
    GERMEVAL_CONFIG,
    MIM_GOLD_NER_CONFIG,
    NORNE_NB_CONFIG,
    NORNE_NN_CONFIG,
    SUC3_CONFIG,
    WIKIANN_FO_CONFIG,
)
from scandeval.named_entity_recognition import NamedEntityRecognition


@pytest.fixture(
    scope="module",
    params=[
        DANE_CONFIG,
        SUC3_CONFIG,
        NORNE_NB_CONFIG,
        NORNE_NN_CONFIG,
        MIM_GOLD_NER_CONFIG,
        WIKIANN_FO_CONFIG,
        GERMEVAL_CONFIG,
        CONLL_NL_CONFIG,
    ],
    ids=[
        "dane",
        "suc3",
        "norne_nb",
        "norne_nn",
        "mim-gold-ner",
        "wikiann-fo",
        "germeval",
        "conll-nl",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield NamedEntityRecognition(
        dataset_config=request.param,
        benchmark_config=benchmark_config,
    )


def test_encoder_sequence_classification(benchmark_dataset, model_id):
    benchmark_dataset.benchmark(model_id)


def test_decoder_sequence_classification(benchmark_dataset, generative_model_id):
    benchmark_dataset.benchmark(generative_model_id)
