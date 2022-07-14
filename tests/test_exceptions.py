"""Unit tests for the `exceptions` module."""

import pytest

from src.scandeval.exceptions import InvalidBenchmark


def test_invalid_benchmark():
    with pytest.raises(InvalidBenchmark):
        raise InvalidBenchmark("Invalid benchmark")
