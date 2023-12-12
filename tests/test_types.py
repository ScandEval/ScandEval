"""Unit tests for the `types` module."""

import typing
from typing import Any, Generator

import pytest
import scandeval.types as types


@pytest.fixture(scope="module")
def module_variable_names() -> Generator[Any, None, None]:
    yield [var for var in dir(types) if "_" not in var and not hasattr(typing, var)]


def test_variable_names_are_upper_case(module_variable_names) -> None:
    for var in module_variable_names:
        assert var.isupper()


def test_variables_subclass_typing_types(module_variable_names) -> None:
    for var in module_variable_names:
        assert hasattr(typing, type(var).__name__)
