"""Functions related to getting the model configuration."""

import logging

from .config import BenchmarkConfig, ModelConfig
from .model_setups import MODEL_SETUP_CLASSES

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
            If the model doesn't exist.
    """
    for setup_class in MODEL_SETUP_CLASSES:
        setup = setup_class(benchmark_config=benchmark_config)
        if setup.model_exists(model_id=model_id):
            return setup.get_model_config(model_id=model_id)
    else:
        raise RuntimeError(f"Model {model_id} not found.")
