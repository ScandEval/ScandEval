"""Functions related to getting the model configuration."""

from .config import BenchmarkConfig, ModelConfig
from .exceptions import InvalidBenchmark
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
            return setup.get_model_config(model_id=model_id)
    else:
        raise InvalidBenchmark(f"Model {model_id} not found.")
