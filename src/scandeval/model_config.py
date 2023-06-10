"""Functions related to getting the model configuration."""

import logging

from .config import BenchmarkConfig, ModelConfig
from .fresh_models import get_fresh_model_config, model_exists_fresh
from .hf_models import get_hf_model_config, model_exists_on_hf_hub
from .local_models import get_local_model_config, model_exists_locally
from .openai_models import get_openai_model_config, model_exists_on_openai

logger = logging.getLogger(__name__)


def get_model_config(model_id: str, benchmark_config: BenchmarkConfig) -> ModelConfig:
    """Fetches configuration for a model.

    Args:
        model_id (str):
            The model ID.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        RuntimeError:
            If the extracted framework is not recognized.
    """
    if model_exists_fresh(model_id=model_id):
        return get_fresh_model_config(model_id=model_id)
    elif model_exists_locally(model_id=model_id):
        return get_local_model_config(
            model_id=model_id,
            framework=benchmark_config.framework,
            raise_errors=benchmark_config.raise_errors,
        )
    elif model_exists_on_hf_hub(
        model_id=model_id, use_auth_token=benchmark_config.use_auth_token
    ):
        return get_hf_model_config(
            model_id=model_id, use_auth_token=benchmark_config.use_auth_token
        )
    elif model_exists_on_openai(
        model_id=model_id, openai_api_key=benchmark_config.openai_api_key
    ):
        return get_openai_model_config(model_id=model_id)
    else:
        raise RuntimeError(
            f"Model {model_id} not found in any of the supported locations."
        )
