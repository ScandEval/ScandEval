"""Unit tests for the `utils` module."""

import random

import numpy as np
import pytest
import torch
from scandeval.utils import (
    enforce_reproducibility,
    is_module_installed,
    should_prompts_be_stripped,
)
from transformers import AutoTokenizer


class TestEnforceReproducibility:
    """Unit tests for the `enforce_reproducibility` function."""

    def test_random_arrays_not_equal(self):
        """Test that two random arrays are not equal."""
        first_random_number = random.random()
        second_random_number = random.random()
        assert first_random_number != second_random_number

    def test_random_arrays_equal(self):
        """Test that two random arrays are equal after enforcing reproducibility."""
        enforce_reproducibility(framework="random")
        first_random_number = random.random()
        enforce_reproducibility(framework="random")
        second_random_number = random.random()
        assert first_random_number == second_random_number

    def test_numpy_arrays_not_equal(self):
        """Test that two random numpy arrays are not equal."""
        first_random_numbers = np.random.rand(10)
        second_random_numbers = np.random.rand(10)
        assert not np.array_equal(first_random_numbers, second_random_numbers)

    def test_numpy_arrays_equal(self):
        """Test that two random arrays are equal after enforcing reproducibility."""
        enforce_reproducibility(framework="numpy")
        first_random_numbers = np.random.rand(10)
        enforce_reproducibility(framework="numpy")
        second_random_numbers = np.random.rand(10)
        assert np.array_equal(first_random_numbers, second_random_numbers)

    def test_pytorch_tensors_not_equal(self):
        """Test that two random pytorch tensors are not equal."""
        first_random_numbers = torch.rand(10)
        second_random_numbers = torch.rand(10)
        assert not torch.equal(first_random_numbers, second_random_numbers)

    def test_pytorch_tensors_equal(self):
        """Test that two random tensors are equal after enforcing reproducibility."""
        enforce_reproducibility(framework="pytorch")
        first_random_numbers = torch.rand(10)
        enforce_reproducibility(framework="pytorch")
        second_random_numbers = torch.rand(10)
        assert torch.equal(first_random_numbers, second_random_numbers)


@pytest.mark.parametrize(
    argnames=["module_name", "expected"],
    argvalues=[
        ("torch", True),
        ("non_existent_module", False),
    ],
    ids=[
        "torch",
        "non_existent_module",
    ],
)
def test_module_is_installed(module_name, expected):
    """Test that a module is installed."""
    assert is_module_installed(module_name) == expected


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[
        ("mistralai/Mistral-7B-v0.1", True),
        ("AI-Sweden-Models/gpt-sw3-6.7b-v2", True),
        ("bert-base-uncased", False),
    ],
)
def test_should_prompts_be_stripped(model_id, expected):
    """Test that a model ID is a generative model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    labels = ["positiv", "negativ"]
    strip_prompts = should_prompts_be_stripped(
        labels_to_be_generated=labels,
        tokenizer=tokenizer,
    )
    assert strip_prompts == expected
