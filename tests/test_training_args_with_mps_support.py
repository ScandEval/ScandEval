"""Unit tests for the `training_args_with_mps_support` module."""

import torch

from scandeval.training_args_with_mps_support import TrainingArgumentsWithMPSSupport


def test_cuda_has_priority():
    """Test that CUDA has priority over MPS and CPU."""
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.set_device = lambda _: None
    torch.backends.mps.is_available = lambda: True
    args = TrainingArgumentsWithMPSSupport(".")
    assert args.device.type == "cuda"


def test_mps_has_second_priority():
    """Test that MPS has priority over CPU."""
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: True
    args = TrainingArgumentsWithMPSSupport(".")
    assert args.device.type == "mps"


def test_cpu_has_third_priority():
    """Test that CPU has the lowest priority."""
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: False
    args = TrainingArgumentsWithMPSSupport(".")
    assert args.device.type == "cpu"
