"""Unit tests for the `model_setups.hf` module."""

import pytest
import torch
from scandeval.model_setups.hf import HFModelSetup


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[
        ("microsoft/mdeberta-v3-base", None),
        ("jonfd/electra-small-nordic", None),
        ("mistralai/Mistral-7B-v0.1", torch.bfloat16),
    ],
    ids=[
        "mdeberta-v3-base",
        "electra-small-nordic",
        "mistral-7b",
    ],
)
def test_torch_dtype_is_set_correctly(benchmark_config, model_id, expected):
    model_setup = HFModelSetup(benchmark_config=benchmark_config)
    hf_model_config = model_setup._load_hf_model_config(
        model_id=model_id,
        num_labels=0,
        id2label=dict(),
        label2id=dict(),
        revision="main",
        model_cache_dir=benchmark_config.cache_dir,
    )
    assert hf_model_config.torch_dtype == expected
