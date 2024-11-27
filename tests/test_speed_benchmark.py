"""Unit tests for the `speed_benchmark` module."""

from typing import Generator

import pytest
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.auto import AutoModelForSequenceClassification, AutoTokenizer

from scandeval.config import ModelConfig
from scandeval.model_setups import HFModelSetup
from scandeval.speed_benchmark import benchmark_speed


@pytest.fixture(scope="module")
def tokenizer(model_id) -> Generator[PreTrainedTokenizer, None, None]:
    """Yields a tokenizer."""
    yield AutoTokenizer.from_pretrained(model_id)


@pytest.fixture(scope="module")
def model(model_id) -> Generator[PreTrainedModel, None, None]:
    """Yields a model."""
    yield AutoModelForSequenceClassification.from_pretrained(model_id)


@pytest.fixture(scope="module")
def model_config(model_id, benchmark_config) -> Generator[ModelConfig, None, None]:
    """Yields a model configuration."""
    model_setup = HFModelSetup(benchmark_config=benchmark_config)
    yield model_setup.get_model_config(model_id=model_id)


class TestBenchmarkSpeed:
    """Unit tests for the `benchmark_speed` function."""

    @pytest.fixture(scope="class")
    def itr(self) -> Generator[tqdm, None, None]:
        """Yields an iterator with a progress bar."""
        yield tqdm(range(2))

    @pytest.fixture(scope="class")
    def scores(
        self, itr, tokenizer, model, model_config, dataset_config, benchmark_config
    ) -> Generator[dict[str, list[dict[str, float]]], None, None]:
        """Yields the benchmark speed scores."""
        yield benchmark_speed(
            itr=itr,
            tokenizer=tokenizer,
            model=model,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

    def test_scores_is_dict(self, scores):
        """Tests that the scores are a dict."""
        assert isinstance(scores, dict)

    def test_scores_keys(self, scores):
        """Tests that the scores have the correct keys."""
        assert set(scores.keys()) == {"test"}

    def test_test_scores_is_list(self, scores):
        """Tests that the test scores are a list."""
        assert isinstance(scores["test"], list)

    def test_test_scores_contain_dicts(self, scores):
        """Tests that the test scores contain dicts."""
        assert all(isinstance(x, dict) for x in scores["test"])

    def test_test_scores_dicts_keys(self, scores):
        """Tests that the test scores dicts have the correct keys."""
        assert all(
            set(x.keys()) == {"test_speed", "test_speed_short"} for x in scores["test"]
        )

    def test_test_scores_dicts_values_dtypes(self, scores):
        """Tests that the test scores dicts have the correct values dtypes."""
        assert all(
            all(isinstance(value, float) for value in x.values())
            for x in scores["test"]
        )
