"""Unit tests for the `utils` module."""

import random

import numpy as np
import torch
from scandeval.utils import enforce_reproducibility, is_module_installed


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


class TestIsModuleInstalled:
    """Unit tests for the `is_module_installed` function."""

    def test_module_is_installed(self):
        """Test that a module is installed."""
        assert is_module_installed("torch")

    def test_module_is_not_installed(self):
        """Test that a module is not installed."""
        assert not is_module_installed("non_existent_module")
