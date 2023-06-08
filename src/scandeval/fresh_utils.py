"""Utility functions related to fresh models."""

from .config import ModelConfig


def get_fresh_model_config(model_id: str) -> ModelConfig:
    """Fetches configuration for a fresh model.

    Args:
        model_id (str):
            The full Hugging Face Hub ID of the model.

    Returns:
        ModelConfig:
            The model configuration.
    """
    return ModelConfig(
        model_id=model_id,
        framework="pytorch",
        task="fill-mask",
        languages=list(),
        revision="main",
    )
