"""Unit tests for the `question_answering` module."""

from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import (
    GERMANQUAD_CONFIG,
    NORQUAD_CONFIG,
    NQII_CONFIG,
    SCANDIQA_DA_CONFIG,
    SCANDIQA_SV_CONFIG,
    SQUAD_CONFIG,
    SQUAD_NL_CONFIG,
)
from scandeval.exceptions import InvalidBenchmark
from scandeval.question_answering import QuestionAnswering, prepare_train_examples
from scandeval.utils import GENERATIVE_DATASET_TASKS
from transformers import AutoTokenizer


@pytest.fixture(
    scope="module",
    params=[
        SCANDIQA_DA_CONFIG,
        SCANDIQA_SV_CONFIG,
        NORQUAD_CONFIG,
        NQII_CONFIG,
        GERMANQUAD_CONFIG,
        SQUAD_CONFIG,
        SQUAD_NL_CONFIG,
    ],
    ids=[
        "scandiqa-da",
        "scandiqa-sv",
        "norquad",
        "nqii",
        "germanquad",
        "squad",
        "squad-nl",
    ],
)
def benchmark_dataset(
    benchmark_config, request
) -> Generator[BenchmarkDataset, None, None]:
    yield QuestionAnswering(
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


@pytest.mark.parametrize(
    argnames="tokenizer_model_id",
    argvalues=[
        "jonfd/electra-small-nordic",
        "flax-community/swe-roberta-wiki-oscar",
    ],
)
@pytest.mark.parametrize(
    argnames="examples",
    argvalues=[
        dict(
            question=["Hvad er hovedstaden i Sverige?"],
            context=["Sveriges hovedstad er Stockholm."],
            answers=[dict(text=["Stockholm"], answer_start=[22])],
        ),
        dict(
            question=["Hvad er hovedstaden i Sverige?"],
            context=["Sveriges hovedstad er Stockholm." * 100],
            answers=[dict(text=["Sverige"], answer_start=[0])],
        ),
        dict(
            question=["Hvad er hovedstaden i Danmark?"],
            context=["Danmarks hovedstad er København. " * 100],
            answers=[dict(text=["Da"], answer_start=[0])],
        ),
    ],
)
def test_prepare_train_examples(examples, tokenizer_model_id):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
    if (
        hasattr(tokenizer, "model_max_length")
        and tokenizer.model_max_length > 100_000_000
    ):
        tokenizer.model_max_length = 512
    with does_not_raise():
        prepare_train_examples(examples=examples, tokenizer=tokenizer)
