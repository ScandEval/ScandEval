"""Unit tests for the `hf_hub` module."""

import pytest

from scandeval.config import ModelConfig
from scandeval.exceptions import InvalidBenchmark
from scandeval.hf_hub import get_model_config, get_model_lists
from scandeval.languages import DA, EN


class TestGetModelConfig:
    @pytest.fixture(scope="class")
    def model_config(self, benchmark_config):
        yield get_model_config(
            model_id="bert-base-uncased", benchmark_config=benchmark_config
        )

    def test_model_config_is_model_config(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_model_config_has_correct_information(self, model_config):
        assert model_config.model_id == "bert-base-uncased"
        assert model_config.revision == "main"
        assert model_config.framework == "pytorch"
        assert model_config.task == "fill-mask"
        assert model_config.languages == [EN]

    def test_invalid_model_id(self, benchmark_config):
        with pytest.raises(InvalidBenchmark):
            get_model_config(
                model_id="invalid-model-id", benchmark_config=benchmark_config
            )

    def test_fresh_model_id(self, benchmark_config):
        model_config = get_model_config(
            model_id="fresh-model-id", benchmark_config=benchmark_config
        )
        assert model_config.model_id == "fresh-model-id"
        assert model_config.revision == "main"
        assert model_config.framework == "pytorch"
        assert model_config.task == "fill-mask"
        assert model_config.languages == []


class TestGetModelListsLanguages:
    @pytest.fixture(scope="class")
    def model_dict(self):
        yield get_model_lists(languages=[DA], use_auth_token=False)

    def test_dict_contains_correct_keys(self, model_dict):
        assert set(model_dict.keys()) == {"da", "all", "multilingual", "fresh"}

    def test_dict_has_non_trivial_values(self, model_dict):
        for val in model_dict.values():
            assert len(val) > 0

    def test_dict_has_no_duplicates(self, model_dict):
        for val in model_dict.values():
            assert len(set(val)) == len(val)

    def test_all_is_the_same_as_the_rest(self, model_dict):
        all_model_ids = {
            model_id
            for key, model_list in model_dict.items()
            for model_id in model_list
            if key != "all"
        }
        assert len(model_dict["all"]) == len(all_model_ids)
