"""Unit tests for the `hf` module."""

import pytest
import torch

from scandeval.benchmark_modules.hf import get_torch_dtype


@pytest.mark.parametrize(
    argnames=["test_device", "torch_dtype_is_set", "bf16_available", "expected"],
    argvalues=[
        ("cpu", True, True, torch.float32),
        ("cpu", True, False, torch.float32),
        ("cpu", False, True, torch.float32),
        ("cpu", False, False, torch.float32),
        ("mps", True, True, torch.float32),
        ("mps", True, False, torch.float32),
        ("mps", False, True, torch.float32),
        ("mps", False, False, torch.float32),
        ("cuda", True, True, "auto"),
        ("cuda", True, False, "auto"),
        ("cuda", False, True, torch.bfloat16),
        ("cuda", False, False, torch.float16),
    ],
)
def test_get_torch_dtype(test_device, torch_dtype_is_set, bf16_available, expected):
    """Test that the torch dtype is set correctly."""
    assert (
        get_torch_dtype(
            device=torch.device(test_device),
            torch_dtype_is_set=torch_dtype_is_set,
            bf16_available=bf16_available,
        )
        == expected
    )
