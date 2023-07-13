"""Unit tests for the `utils` module."""

import random

import numpy as np
import torch

from scandeval.utils import enforce_reproducibility, is_module_installed


class TestEnforceReproducibility:
    def test_random_arrays_not_equal(self):
        first_random_number = random.random()
        second_random_number = random.random()
        assert first_random_number != second_random_number

    def test_random_arrays_equal(self):
        enforce_reproducibility(framework="random")
        first_random_number = random.random()
        enforce_reproducibility(framework="random")
        second_random_number = random.random()
        assert first_random_number == second_random_number

    def test_numpy_arrays_not_equal(self):
        first_random_numbers = np.random.rand(10)
        second_random_numbers = np.random.rand(10)
        assert not np.array_equal(first_random_numbers, second_random_numbers)

    def test_numpy_arrays_equal(self):
        enforce_reproducibility(framework="numpy")
        first_random_numbers = np.random.rand(10)
        enforce_reproducibility(framework="numpy")
        second_random_numbers = np.random.rand(10)
        assert np.array_equal(first_random_numbers, second_random_numbers)

    def test_pytorch_tensors_not_equal(self):
        first_random_numbers = torch.rand(10)
        second_random_numbers = torch.rand(10)
        assert not torch.equal(first_random_numbers, second_random_numbers)

    def test_pytorch_tensors_equal(self):
        enforce_reproducibility(framework="pytorch")
        first_random_numbers = torch.rand(10)
        enforce_reproducibility(framework="pytorch")
        second_random_numbers = torch.rand(10)
        assert torch.equal(first_random_numbers, second_random_numbers)


class TestIsModuleInstalled:
    def test_module_is_installed(self):
        assert is_module_installed("torch")

    def test_module_is_not_installed(self):
        assert not is_module_installed("non_existent_module")
