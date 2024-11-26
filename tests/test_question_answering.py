"""Unit tests for the `question_answering` module."""

import os
from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from transformers import AutoTokenizer

from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.exceptions import InvalidBenchmark
from scandeval.languages import DA
from scandeval.question_answering import QuestionAnswering, prepare_train_examples
from scandeval.tasks import RC
from scandeval.utils import GENERATIVE_DATASET_TASKS


@pytest.fixture(
    scope="module",
    params=[
        dataset_config
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config.task == RC
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
    """Yields a question answering benchmark dataset."""
    yield QuestionAnswering(
        dataset_config=request.param, benchmark_config=benchmark_config
    )


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
def test_encoder_benchmarking(benchmark_dataset, model_id):
    """Test that encoder models can be benchmarked on question answering datasets."""
    if benchmark_dataset.dataset_config.task.name in GENERATIVE_DATASET_TASKS:
        with pytest.raises(InvalidBenchmark):
            benchmark_dataset.benchmark(model_id)
    else:
        with does_not_raise():
            benchmark_dataset.benchmark(model_id)


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
def test_decoder_benchmarking(
    benchmark_dataset, generative_model_id, generative_model_and_tokenizer
):
    """Test that decoder models can be benchmarked on question answering datasets."""
    model, tokenizer = generative_model_and_tokenizer
    with does_not_raise():
        benchmark_dataset.benchmark(
            model_id=generative_model_id, model=model, tokenizer=tokenizer
        )


@pytest.mark.parametrize(
    argnames="tokenizer_model_id",
    argvalues=["jonfd/electra-small-nordic", "flax-community/swe-roberta-wiki-oscar"],
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
    """Test that train examples can be prepared for training."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
    if (
        hasattr(tokenizer, "model_max_length")
        and tokenizer.model_max_length > 100_000_000
    ):
        tokenizer.model_max_length = 512
    with does_not_raise():
        prepare_train_examples(examples=examples, tokenizer=tokenizer)
