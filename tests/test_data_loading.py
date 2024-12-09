"""Tests for the `data_loading` module."""

import time
from collections.abc import Generator

import pytest
from datasets import DatasetDict
from numpy.random import default_rng

from scandeval.data_loading import load_data
from scandeval.dataset_configs import SPEED_CONFIG, get_all_dataset_configs
from scandeval.exceptions import HuggingFaceHubDown


@pytest.mark.parametrize(
    argnames=["dataset_config"],
    argvalues=[
        (dataset_config,)
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config != SPEED_CONFIG
    ],
    ids=[
        (dataset_config.name)
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config != SPEED_CONFIG
    ],
    scope="class",
)
class TestLoadData:
    """Tests for the `load_data` function."""

    @pytest.fixture(scope="class")
    def dataset(
        self, dataset_config, benchmark_config
    ) -> Generator[list[DatasetDict], None, None]:
        """A loaded dataset."""
        for _ in range(10):
            try:
                yield load_data(
                    rng=default_rng(seed=4242),
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                )
                break
            except HuggingFaceHubDown:
                time.sleep(5)
        else:
            raise HuggingFaceHubDown()

    def test_load_data_is_list_of_dataset_dicts(self, dataset):
        """Test that the `load_data` function returns a list of `DatasetDict`."""
        assert isinstance(dataset, list)
        assert all(isinstance(d, DatasetDict) for d in dataset)

    def test_split_names_are_correct(self, dataset):
        """Test that the split names are correct."""
        assert all(set(d.keys()) == {"train", "val", "test"} for d in dataset)

    def test_number_of_iterations_is_correct(self, dataset, benchmark_config):
        """Test that the number of iterations is correct."""
        assert len(dataset) == benchmark_config.num_iterations

    def test_no_empty_examples(self, dataset):
        """Test that there are no empty examples in the dataset."""
        for dataset in dataset:
            for split in dataset.values():
                for feature in ["text", "tokens"]:
                    if feature in split.features:
                        assert all(len(x) > 0 for x in split[feature])
