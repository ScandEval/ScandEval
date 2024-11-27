"""Unit tests for the `model_setups.hf` module."""

import copy

import pytest
import torch

from scandeval.model_setups.hf import HFModelSetup


@pytest.mark.parametrize(
    argnames=["test_device", "model_id", "expected"],
    argvalues=[
        ("cuda", "microsoft/mdeberta-v3-base", torch.float16),
        ("cuda", "jonfd/electra-small-nordic", torch.float16),
        ("cuda", "mistralai/Mistral-7B-v0.1", "auto"),
        ("cuda", "microsoft/phi-2", "auto"),
        ("mps", "microsoft/mdeberta-v3-base", torch.float32),
        ("mps", "jonfd/electra-small-nordic", torch.float32),
        ("mps", "mistralai/Mistral-7B-v0.1", torch.float32),
        ("mps", "microsoft/phi-2", torch.float32),
        ("cpu", "microsoft/mdeberta-v3-base", torch.float32),
        ("cpu", "jonfd/electra-small-nordic", torch.float32),
        ("cpu", "mistralai/Mistral-7B-v0.1", torch.float32),
        ("cpu", "microsoft/phi-2", torch.float32),
    ],
    ids=[
        "cuda:mdeberta-v3-base",
        "cuda:electra-small-nordic",
        "cuda:mistral-7b",
        "cuda:phi-2",
        "mps:mdeberta-v3-base",
        "mps:electra-small-nordic",
        "mps:mistral-7b",
        "mps:phi-2",
        "cpu:mdeberta-v3-base",
        "cpu:electra-small-nordic",
        "cpu:mistral-7b",
        "cpu:phi-2",
    ],
)
def test_torch_dtype_is_set_correctly(
    benchmark_config, test_device, model_id, expected
):
    """Test that the torch dtype is set correctly."""
    benchmark_config_copy = copy.deepcopy(benchmark_config)
    benchmark_config_copy.device = torch.device(test_device)
    model_setup = HFModelSetup(benchmark_config=benchmark_config_copy)
    hf_model_config = model_setup._load_hf_model_config(
        model_id=model_id,
        num_labels=0,
        id2label=dict(),
        label2id=dict(),
        revision="main",
        model_cache_dir=benchmark_config_copy.cache_dir,
    )

    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if test_device == "cuda" and expected == torch.float16 and bf16_available:
        expected = torch.bfloat16

    assert model_setup._get_torch_dtype(config=hf_model_config) == expected
