"""Unit tests for the `exceptions` module."""

import pytest

from scandeval.exceptions import (
    HuggingFaceHubDown,
    InvalidBenchmark,
    NoInternetConnection,
)


def test_invalid_benchmark_is_an_exception():
    with pytest.raises(InvalidBenchmark):
        raise InvalidBenchmark()


def test_hugging_face_hub_down_is_an_exception():
    with pytest.raises(HuggingFaceHubDown):
        raise HuggingFaceHubDown()


def test_no_internet_connection_is_an_exception():
    with pytest.raises(NoInternetConnection):
        raise NoInternetConnection()
