"""Unit tests for the `dataset_configs` module."""

import pytest
from scandeval.config import DatasetConfig
from scandeval.dataset_configs import get_all_dataset_configs, get_dataset_config


class TestGetAllDatasetConfigs:
    @pytest.fixture(scope="class")
    def dataset_configs(self):
        yield get_all_dataset_configs()

    def test_dataset_configs_is_dict(self, dataset_configs):
        assert isinstance(dataset_configs, dict)

    def test_dataset_configs_are_objects(self, dataset_configs):
        for dataset_config in dataset_configs.values():
            assert isinstance(dataset_config, DatasetConfig)


class TestGetDatasetConfig:
    def test_get_angry_tweets_config(self):
        dataset_config = get_dataset_config("angry-tweets")
        assert dataset_config.name == "angry-tweets"

    def test_error_when_dataset_does_not_exist(self):
        with pytest.raises(ValueError):
            get_dataset_config("does-not-exist")
