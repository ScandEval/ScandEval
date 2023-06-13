"""Model setup for OpenAI models."""

import logging
import os

import openai
import tiktoken
from transformers import PretrainedConfig, PreTrainedModel

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig
from ..enums import Framework, ModelType
from ..openai_models import OpenAIModel, OpenAITokenizer
from .base import GenerativeModel, Tokenizer

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


VOCAB_SIZE_MAPPING = {
    "ada": 50_257,
    "text-ada-001": 50_257,
    "babbage": 50_257,
    "curie": 50_257,
    "davinci": 50_257,
    "text-babbage-001": 50_257,
    "text-curie-001": 50_257,
    "text-davinci-001": 50_257,
    "text-davinci-002": 50_281,
    "text-davinci-003": 50_281,
    "code-davinci-001": 50_281,
    "code-davinci-002": 50_281,
    "gpt-3.5-turbo": 100_256,
    "gpt-3.5-turbo-0301": 100_256,
    "gpt-4": 100_256,
    "gpt-4-0314": 100_256,
    "gpt-4-32k": 100_256,
    "gpt-4-32k-0314": 100_256,
}


MODEL_MAX_LENGTH_MAPPING = {
    "ada": 2049,
    "babbage": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-ada-001": 2049,
    "text-babbage-001": 2049,
    "text-curie-001": 2049,
    "text-davinci-001": 2049,
    "text-davinci-002": 4097,
    "text-davinci-003": 4097,
    "code-davinci-001": 8001,
    "code-davinci-002": 8001,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32_768,
    "gpt-4-32k-0314": 32_768,
}


NUM_PARAMS_MAPPING = {
    "ada": 350_000_000,
    "babbage": 3_000_000_000,
    "curie": 13_000_000_000,
    "davinci": 175_000_000_000,
    "text-ada-001": 350_000_000,
    "text-babbage-001": 3_000_000_000,
    "text-curie-001": 13_000_000_000,
    "text-davinci-001": 175_000_000_000,
    "text-davinci-002": 175_000_000_000,
    "text-davinci-003": 175_000_000_000,
    "code-davinci-001": 175_000_000_000,
    "code-davinci-002": 175_000_000_000,
    "gpt-3.5-turbo": 175_000_000_000,
    "gpt-3.5-turbo-0301": 175_000_000_000,
    "gpt-4": -1,
    "gpt-4-0314": -1,
    "gpt-4-32k": -1,
    "gpt-4-32k-0314": -1,
}


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
    ) -> tuple[Tokenizer, PreTrainedModel | GenerativeModel]:
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
        hf_model_config = PretrainedConfig.from_pretrained("gpt2")
        hf_model_config.vocab_size = (
            VOCAB_SIZE_MAPPING.get(model_config.model_id, -2) + 1
        )
        hf_model_config.model_max_length = MODEL_MAX_LENGTH_MAPPING.get(
            model_config.model_id, -1
        )
        hf_model_config.num_params = NUM_PARAMS_MAPPING.get(model_config.model_id, -1)
        hf_model_config.id2label = dataset_config.id2label
        hf_model_config.label2id = dataset_config.label2id
        hf_model_config.eos_token_id = hf_model_config.vocab_size - 2
        hf_model_config.bos_token_id = hf_model_config.vocab_size - 2
        hf_model_config.pad_token_id = hf_model_config.vocab_size - 1

        # If the vocab size is -1, we're finding it by brute force
        if hf_model_config.vocab_size == -1:
            tok = tiktoken.encoding_for_model(model_name=model_config.model_id)
            for idx in range(1, 100_256, -1):
                try:
                    tok.decode([idx])
                    hf_model_config.vocab_size = idx + 1
                    break
                except KeyError:
                    pass
            else:
                raise ValueError(
                    f"Couldn't find vocab size for model {model_config.model_id}"
                )

        tokenizer = OpenAITokenizer(
            model_config=model_config, hf_model_config=hf_model_config
        )
        model = OpenAIModel(
            model_config=model_config,
            hf_model_config=hf_model_config,
            dataset_config=dataset_config,
            benchmark_config=self.benchmark_config,
            tokenizer=tokenizer,
        )
        return tokenizer, model
