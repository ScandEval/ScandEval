"""Functions related to getting the model configuration."""

import logging
from pathlib import Path

from .config import BenchmarkConfig, ModelConfig

# from .fresh_utils import get_fresh_model_config
from .hf_utils import get_hf_model_config, model_exists_on_hf_hub
from .local_utils import get_local_model_config

# from .openai_utils import get_openai_model_config

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
    if Path(model_id).is_dir():
        model_configurator = get_local_model_config
    elif model_exists_on_hf_hub(
        model_id=model_id, use_auth_token=benchmark_config.use_auth_token
    ):
        model_configurator = get_hf_model_config

    model_config = model_configurator(
        model_id=model_id, benchmark_config=benchmark_config
    )
    return model_config
