"""Model setup for OpenAI models."""

import logging
import os

import openai
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig
from ..enums import Framework, ModelType

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


class OpenAIModelSetup:
    """Model setup for OpenAI models.

    Args:
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.

    Attributes:
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
    """

    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        self.benchmark_config = benchmark_config

    def model_exists(self, model_id: str) -> bool:
        """Check if a model ID denotes an OpenAI model.

        Args:
            model_id (str):
                The model ID.

        Returns:
            bool:
                Whether the model exists on OpenAI.
        """
        if self.benchmark_config.openai_api_key is not None:
            openai.api_key = self.benchmark_config.openai_api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if openai.api_key is not None:
            all_models = openai.Model.list()["data"]
            return model_id in [model["id"] for model in all_models]
        else:
            model_exists = model_id in CACHED_OPENAI_MODEL_IDS
            if model_exists:
                logger.warning(
                    "It looks like you're trying to use an OpenAI model, but you "
                    "haven't set your OpenAI API key. Please set your OpenAI API key "
                    "using the environment variable `OPENAI_API_KEY`, or by passing it "
                    "as the `--openai-api-key` argument."
                )
            else:
                logger.info(
                    "It doesn't seem like the model exists on OpenAI, but we can't be "
                    "sure because you haven't set your OpenAI API key. If you intended "
                    "to use an OpenAI model, please set your OpenAI API key using the "
                    "environment variable `OPENAI_API_KEY`, or by passing it as the "
                    "`--openai-api-key` argument."
                )
            return model_exists

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Fetches configuration for an OpenAI model.

        Args:
            model_id (str):
                The model ID of the model.

        Returns:
            ModelConfig:
                The model configuration.
        """
        return ModelConfig(
            model_id=model_id,
            revision="main",
            framework=Framework.API,
            task="text-generation",
            languages=list(),
            model_type=ModelType.OPENAI,
        )

    def load_model(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Load an OpenAI model.

        Args:
            model_config (ModelConfig):
                The model configuration.
            dataset_config (DatasetConfig):
                The dataset configuration.

        Returns:
            pair of (tokenizer, model):
                The tokenizer and model.
        """
        raise NotImplementedError
