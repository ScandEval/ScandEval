"""Utility functions related to OpenAI models."""

import logging

import openai

from .config import ModelConfig
from .enums import Framework
from .languages import get_all_languages

logger = logging.getLogger(__name__)


def get_openai_model_config(model_id: str) -> ModelConfig:
    """Fetches configuration for a model from the Hugging Face Hub.

    Args:
        model_id (str):
            The OpenAI model ID.

    Returns:
        ModelConfig:
            The model configuration.
    """
    return ModelConfig(
        model_id=model_id,
        framework=Framework.OPENAI,
        task="text-generation",
        languages=list(get_all_languages().values()),
        revision="main",
    )


def model_exists_on_openai(model_id: str, openai_api_key: str | None) -> bool | None:
    """Checks whether a model exists on OpenAI.

    Args:
        model_id (str):
            The model ID to check.
        openai_api_key (str or None):
            The OpenAI API key to use for authentication. If None, then None will be
            returned.

    Returns:
        bool or None:
            If model exists on OpenAI or not. If the API key is None, then None will
            be returned.
    """
    if openai_api_key is None:
        return None
    openai.api_key = openai_api_key
    model_list = openai.Model.list()
    return model_id in [model["id"] for model in model_list["data"]]
