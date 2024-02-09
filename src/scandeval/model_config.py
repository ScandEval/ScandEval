"""Functions related to getting the model configuration."""

import importlib.util

from .config import BenchmarkConfig, ModelConfig
from .enums import Framework
from .exceptions import InvalidModel, NeedsExtraInstalled
from .model_setups import MODEL_SETUP_CLASSES


def get_model_config(model_id: str, benchmark_config: BenchmarkConfig) -> ModelConfig:
    """Fetches configuration for a model.

    Args:
        model_id:
            The model ID.
        benchmark_config:
            The configuration of the benchmark.

    Returns:
        The model configuration.

    Raises:
        RuntimeError:
            If the model doesn't exist.
    """
    for setup_class in MODEL_SETUP_CLASSES:
        setup = setup_class(benchmark_config=benchmark_config)
        if setup.model_exists(model_id=model_id):
            model_config = setup.get_model_config(model_id=model_id)
            if (
                model_config.framework == Framework.JAX
                and importlib.util.find_spec("jax") is None
            ):
                raise NeedsExtraInstalled(extra="jax")
            return model_config
    else:
        raise InvalidModel(f"Model {model_id} not found.")
