"""Unit tests for the `generation` module."""

from typing import Generator

import pytest
from scandeval.generation import (
    GenerativeModelOutput,
)


@pytest.fixture(scope="module")
def cache() -> Generator[dict[str, dict[str, GenerativeModelOutput]], None, None]:
    yield dict()


def test_load_model_cache():
    pass


def test_store_cache_to_disk():
    pass


def test_split_dataset_into_cached_and_non_cached():
    pass


def test_load_cached_model_outputs():
    pass


def test_extract_raw_predictions():
    pass


def test_get_generation_stopping_criteria():
    pass


def test_stop_word_criteria():
    pass
