"""Unit tests for the `named_entity_recognition` module."""

import os
from contextlib import nullcontext as does_not_raise
from typing import Generator

import pytest
from scandeval.benchmark_dataset import BenchmarkDataset
from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.exceptions import InvalidBenchmark
from scandeval.languages import DA
from scandeval.model_config import get_model_config
from scandeval.model_loading import load_model
from scandeval.named_entity_recognition import NamedEntityRecognition
from scandeval.protocols import GenerativeModel, Tokenizer
from scandeval.tasks import NER
from scandeval.utils import GENERATIVE_DATASET_TASKS


@pytest.fixture(
    scope="module",
    params=[
        dataset_config
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config.task == NER
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
    """Yields a named entity recognition dataset."""
    yield NamedEntityRecognition(
        dataset_config=request.param, benchmark_config=benchmark_config
    )


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
def test_encoder_benchmarking(benchmark_dataset, model_id):
    """Test that encoder models can be benchmarked on named entity recognition."""
    if benchmark_dataset.dataset_config.task.name in GENERATIVE_DATASET_TASKS:
        with pytest.raises(InvalidBenchmark):
            benchmark_dataset.benchmark(model_id)
    else:
        with does_not_raise():
            benchmark_dataset.benchmark(model_id)


@pytest.mark.skipif(condition=os.getenv("TEST_EVALUATIONS") == "0", reason="Skipped")
class TestDecoderBenchmarking:
    """Test that decoder models can be benchmarked on named entity recognition."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(
        self, benchmark_config, generative_model_id, benchmark_dataset
    ) -> Generator[tuple[GenerativeModel, Tokenizer], None, None]:
        """Yields a model and tokenizer."""
        model_config = get_model_config(
            model_id=generative_model_id, benchmark_config=benchmark_config
        )
        model, tokenizer = load_model(
            model_config=model_config,
            dataset_config=benchmark_dataset.dataset_config,
            benchmark_config=benchmark_config,
        )
        yield model, tokenizer

    def test_decoder_benchmarking(
        self, benchmark_dataset, generative_model_id, model_and_tokenizer
    ):
        """Test that decoder models can be benchmarked on named entity recognition."""
        model, tokenizer = model_and_tokenizer
        with does_not_raise():
            benchmark_dataset.benchmark(
                model_id=generative_model_id, model=model, tokenizer=tokenizer
            )
