"""Functions related to the existence of OpenAI models."""

import logging
import os

import openai

logger = logging.getLogger(__name__)


# This is a list of all OpenAI language models available as of June 10, 2023. It's used
# to check if a model ID denotes an OpenAI model, without having to use an OpenAI API
# key
CACHED_OPENAI_MODEL_IDS: list[str] = [
    "babbage",
    "davinci",
    "text-davinci-edit-001",
    "babbage-code-search-code",
    "text-similarity-babbage-001",
    "code-davinci-edit-001",
    "text-davinci-001",
    "ada",
    "babbage-code-search-text",
    "babbage-similarity",
    "code-search-babbage-text-001",
    "text-curie-001",
    "code-search-babbage-code-001",
    "text-ada-001",
    "text-similarity-ada-001",
    "curie-instruct-beta",
    "ada-code-search-code",
    "ada-similarity",
    "code-search-ada-text-001",
    "text-search-ada-query-001",
    "davinci-search-document",
    "ada-code-search-text",
    "text-search-ada-doc-001",
    "davinci-instruct-beta",
    "text-similarity-curie-001",
    "code-search-ada-code-001",
    "ada-search-query",
    "text-search-davinci-query-001",
    "curie-search-query",
    "davinci-search-query",
    "babbage-search-document",
    "ada-search-document",
    "text-search-curie-query-001",
    "text-search-babbage-doc-001",
    "curie-search-document",
    "text-search-curie-doc-001",
    "babbage-search-query",
    "text-babbage-001",
    "text-search-davinci-doc-001",
    "text-embedding-ada-002",
    "text-search-babbage-query-001",
    "curie-similarity",
    "curie",
    "text-similarity-davinci-001",
    "text-davinci-002",
    "gpt-4-0314",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "davinci-similarity",
    "gpt-4",
    "gpt-3.5-turbo-0301",
]


def model_exists_on_openai(model_id: str, openai_api_key: str | None) -> bool:
    """Check if a model exists on OpenAI.

    Args:
        model_id (str or Path):
            Path to the model folder.
        openai_api_key (str or None):
            The OpenAI API key. If None, the environment variable `OPENAI_API_KEY` is
            used.

    Returns:
        bool:
            Whether the model exists on OpenAI.
    """
    openai.api_key = openai_api_key if openai_api_key else os.getenv("OPENAI_API_KEY")
    if openai.api_key is not None:
        all_models = openai.Model.list()["data"]
        return model_id in [model["id"] for model in all_models]
    else:
        model_exists = model_id in CACHED_OPENAI_MODEL_IDS
        if model_exists:
            logger.warning(
                "It looks like you're trying to use an OpenAI model, but you haven't "
                "set your OpenAI API key. Please set your OpenAI API key using the "
                "environment variable OPENAI_API_KEY, or by passing it as the "
                "`--openai-api-key` argument."
            )
        else:
            logger.info(
                "It doesn't seem like the model exists on OpenAI, but we can't be "
                "sure because you haven't set your OpenAI API key. If you intended "
                "to use an OpenAI model, please set your OpenAI API key using the "
                "environment variable OPENAI_API_KEY, or by passing it as the "
                "`--openai-api-key` argument."
            )
        return model_exists
