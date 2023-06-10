"""Functions related to configurations of OpenAI models."""

import logging

from ..config import ModelConfig
from ..enums import Framework

logger = logging.getLogger(__name__)


def get_openai_model_config(model_id: str) -> ModelConfig:
    """Builds the configuration for an OpenAI model.

    Args:
        model_id (str):
            The path of the local model.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Returns:
        ModelConfig:
            The model configuration.
    """
    return ModelConfig(
        model_id=model_id,
        revision="main",
        framework=Framework.OPENAI,
        task="text-generation",
        languages=list(),
    )
