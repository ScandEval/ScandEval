"""Unit tests for the `speed_benchmark` module."""

import pytest
from tqdm.auto import tqdm
from transformers.models.auto import AutoModelForSequenceClassification, AutoTokenizer

from scandeval.dataset_configs import SPEED_CONFIG
from scandeval.model_setups import HFModelSetup
from scandeval.speed_benchmark import benchmark_speed


@pytest.fixture(scope="module")
def tokenizer(model_id):
    yield AutoTokenizer.from_pretrained(model_id)


@pytest.fixture(scope="module")
def model(model_id):
    yield AutoModelForSequenceClassification.from_pretrained(model_id)


@pytest.fixture(scope="module")
def model_config(model_id, benchmark_config):
    model_setup = HFModelSetup(benchmark_config=benchmark_config)
    return model_setup.get_model_config(model_id=model_id)


@pytest.fixture(scope="module")
def dataset_config():
    yield SPEED_CONFIG


class TestBenchmarkSpeed:
    @pytest.fixture(scope="class")
    def itr(self):
        yield tqdm(range(2))

    @pytest.fixture(scope="class")
    def scores(
        self, itr, tokenizer, model, model_config, dataset_config, benchmark_config
    ):
        yield benchmark_speed(
            itr=itr,
            tokenizer=tokenizer,
            model=model,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

    def test_scores_is_dict(self, scores):
        assert isinstance(scores, dict)

    def test_scores_keys(self, scores):
        assert set(scores.keys()) == {"test"}

    def test_test_scores_is_list(self, scores):
        assert isinstance(scores["test"], list)

    def test_test_scores_contain_dicts(self, scores):
        assert all(isinstance(x, dict) for x in scores["test"])

    def test_test_scores_dicts_keys(self, scores):
        assert all(
            set(x.keys()) == {"test_speed", "test_speed_short"} for x in scores["test"]
        )

    def test_test_scores_dicts_values_dtypes(self, scores):
        assert all(
            all(isinstance(value, float) for value in x.values())
            for x in scores["test"]
        )
