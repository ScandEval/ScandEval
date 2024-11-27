"""Unit tests for the `model_loading` module."""

import os

import pytest

from scandeval.config import ModelConfig
from scandeval.enums import Framework
from scandeval.exceptions import InvalidBenchmark, InvalidModel
from scandeval.languages import DA
from scandeval.model_config import get_model_config
from scandeval.model_loading import load_model


def test_load_non_generative_model(model_id, dataset_config, benchmark_config):
    """Test loading a non-generative model."""
    model_config = get_model_config(
        model_id=model_id, benchmark_config=benchmark_config
    )
    tokenizer, model = load_model(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )
    assert tokenizer is not None
    assert model is not None


@pytest.mark.skipif(
    condition=os.getenv("USE_VLLM", "0") != "1", reason="Not using VLLM."
)
def test_load_generative_model(generative_model_and_tokenizer):
    """Test loading a generative model."""
    model, tokenizer = generative_model_and_tokenizer
    assert tokenizer is not None
    assert model is not None


def test_load_non_generative_model_with_generative_data(
    model_id, generative_dataset_config, benchmark_config
):
    """Test loading a non-generative model with generative data."""
    model_config = get_model_config(
        model_id=model_id, benchmark_config=benchmark_config
    )
    with pytest.raises(InvalidBenchmark):
        load_model(
            model_config=model_config,
            dataset_config=generative_dataset_config,
            benchmark_config=benchmark_config,
        )


def test_load_non_existing_model(dataset_config, benchmark_config):
    """Test loading a non-existing model."""
    model_config = ModelConfig(
        model_id="non-existing-model",
        revision="revision",
        framework=Framework.PYTORCH,
        task="task",
        languages=[DA],
        model_type="fresh",
        model_cache_dir="cache_dir",
        adapter_base_model_id=None,
    )
    with pytest.raises(InvalidModel):
        load_model(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )


# TODO: Fix these OOM errors.
@pytest.mark.skip(reason="Skipping quantisation tests due to OOM errors.")
class TestQuantisedModels:
    """Tests for quantised models."""

    def test_load_awq_model(
        self, awq_generative_model_id, dataset_config, benchmark_config
    ):
        """Test loading an AWQ quantised model."""
        if benchmark_config.device.type != "cuda":
            pytest.skip("Skipping test because the device is not a GPU.")
        model_config = get_model_config(
            model_id=awq_generative_model_id, benchmark_config=benchmark_config
        )
        load_model(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

    def test_load_gptq_model(
        self, gptq_generative_model_id, dataset_config, benchmark_config
    ):
        """Test loading a GPTQ quantised model."""
        if benchmark_config.device.type != "cuda":
            pytest.skip("Skipping test because the device is not a GPU.")
        model_config = get_model_config(
            model_id=gptq_generative_model_id, benchmark_config=benchmark_config
        )
        load_model(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
