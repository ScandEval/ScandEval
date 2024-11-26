"""Unit tests for the `types` module."""

import re
import typing
from typing import Generator

import pytest

import scandeval.types as types


@pytest.fixture(scope="module")
def module_variable_names() -> Generator[list[str], None, None]:
    """Yields the module variable names."""
    yield [
        var
        for var in dir(types)
        if "_" not in var and not hasattr(typing, var) and var != "np"
    ]


def test_type_variable_names_are_title_case(module_variable_names) -> None:
    """Tests that all type variable names are title case."""
    for var in module_variable_names:
        assert var[0].isupper()
        assert re.search(r"[^A-Za-z]", var) is None
