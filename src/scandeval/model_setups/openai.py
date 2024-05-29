"""Model setup for OpenAI models."""

import importlib.util
import logging
import re
from typing import TYPE_CHECKING

from transformers import PretrainedConfig

from ..config import ModelConfig
from ..enums import Framework, ModelType
from ..openai_models import OpenAIModel, OpenAITokenizer
from ..utils import create_model_cache_dir

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..config import BenchmarkConfig, DatasetConfig
    from ..protocols import GenerativeModel, Tokenizer

if importlib.util.find_spec("openai") is not None:
    import openai

    # Older versions of `openai` doesn't have the `models` module, so we need to check
    # that, as it will cause errors later otherwise
    openai.models


logger = logging.getLogger(__package__)


# This is a list of the major models that OpenAI has released
CACHED_OPENAI_MODEL_IDS: list[str] = [
    "ada|babbage|curie|davinci",
    "(code|text)-(ada|babbage|curie|davinci)-[0-9]{3}",
    "gpt-3.5-turbo(-16k|-instruct)?(-[0-9]{4})?(-preview)?",
    "gpt-4(-[0-9]{4})?(-preview)?",
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?",
    "gpt-4-32k(-[0-9]{4})?(-preview)?",
]


VOCAB_SIZE_MAPPING = {
    "(text-)?(ada|babbage|curie|davinci)(-001)?": 50_257,
    "(code|text)-davinci-00[2-9]": 50_281,
    "gpt-3.5-turbo(-16k)?(-[0-9]{4})?": 100_256,
    "gpt-4-(32k)?(-[0-9]{4})?": 100_256,
    "gpt-4-[0-9]{4}-preview": 100_256,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 100_256,
    "gpt-4-(vision|turbo)(-preview)?": 100_256,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": 100_256,
    "gpt-4o(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_019,
}


MODEL_MAX_LENGTH_MAPPING = {
    "(text-)?(ada|babbage|curie|davinci)(-001)?": 2_050,
    "text-davinci-00[2-9]": 4_098,
    "code-davinci-00[1-9]": 8_002,
    "gpt-3.5-turbo-0613": 4_096,
    "gpt-3.5-turbo(-[0-9]{4})?": 16_385,
    "gpt-3.5-turbo-16k(-[0-9]{4})?": 16_384,
    "gpt-4(-[0-9]{4})?": 8_191,
    "gpt-4-32k(-[0-9]{4})?": 32_767,
    "gpt-4-[0-9]{4}-preview": 128_000,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    "gpt-4-(vision|turbo)(-preview)?": 128_000,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": 4_095,
    "gpt-4o(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
}


NUM_PARAMS_MAPPING = {
    "(text-)?ada(-001)?": 350_000_000,
    "(text-)?babbage(-001)?": 3_000_000_000,
    "(text-)?curie(-001)?": 13_000_000_000,
    "((text|code)-)?davinci(-00[1-9])?": 175_000_000_000,
    "gpt-(3.5|4)-turbo-((16|32)k)?(-[0-9]{4})?": -1,
    "gpt-4-[0-9]{4}-preview": -1,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    "gpt-4-(vision|turbo)(-preview)?": -1,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": -1,
    "gpt-4o(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
}


class OpenAIModelSetup:
    """Model setup for OpenAI models.

    Attributes:
        benchmark_config:
            The benchmark configuration.
    """

    def __init__(self, benchmark_config: "BenchmarkConfig") -> None:
        """Initialize the model setup.

        Args:
            benchmark_config:
                The benchmark configuration.
        """
        self.benchmark_config = benchmark_config

    def model_exists(self, model_id: str) -> bool | dict[str, str]:
        """Check if a model ID denotes an OpenAI model.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exists, or a dictionary explaining why we cannot check
            whether the model exists.
        """
        if importlib.util.find_spec("openai") is None:
            return dict(missing_extra="openai")

        # The model ID for the Azure OpenAI API is the deployment name and therefore
        # different from the model ID used in the OpenAI API. We'll just assume that
        # the model exists in this case.
        if self.benchmark_config.azure_openai_api_key is not None:
            return True

        all_models: list[openai.models.Model] = list()
        try:
            all_models = list(openai.models.list())
        except openai.OpenAIError as e:
            model_exists = any(
                [
                    re.match(pattern=model_pattern, string=model_id) is not None
                    for model_pattern in CACHED_OPENAI_MODEL_IDS
                ]
            )
            if not model_exists:
                if "OPENAI_API_KEY" in str(e):
                    return dict(missing_env_var="OPENAI_API_KEY")
                elif "AZURE_OPENAI_API_KEY" in str(e):
                    return dict(missing_env_var="AZURE_OPENAI_API_KEY")
                elif "AZURE_OPENAI_ENDPOINT" in str(e):
                    return dict(missing_env_var="AZURE_OPENAI_ENDPOINT")

        return model_id in [model.id for model in all_models]

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Fetches configuration for an OpenAI model.

        Args:
            model_id:
                The model ID of the model.

        Returns:
            The model configuration.
        """
        return ModelConfig(
            model_id=model_id,
            revision="main",
            framework=Framework.API,
            task="text-generation",
            languages=list(),
            model_type=ModelType.OPENAI,
            model_cache_dir=create_model_cache_dir(
                cache_dir=self.benchmark_config.cache_dir, model_id=model_id
            ),
        )

    def load_model(
        self, model_config: ModelConfig, dataset_config: "DatasetConfig"
    ) -> tuple["PreTrainedModel | GenerativeModel", "Tokenizer"]:
        """Load an OpenAI model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.

        Returns:
            The tokenizer and model.
        """
        hf_model_config = PretrainedConfig.from_pretrained("gpt2")

        vocab_sizes = [
            vocab_size
            for pattern, vocab_size in VOCAB_SIZE_MAPPING.items()
            if re.match(pattern=pattern, string=model_config.model_id)
        ]
        hf_model_config.vocab_size = vocab_sizes[0] if vocab_sizes else 100_256

        # We subtract the maximum generation length, as that counts towards the total
        # amount of tokens that the model needs to process.
        # We subtract 1 as errors occur if the model is exactly at the maximum length.
        model_lengths = [
            model_length - dataset_config.max_generated_tokens - 1
            for pattern, model_length in MODEL_MAX_LENGTH_MAPPING.items()
            if re.match(pattern=f"^{pattern}$", string=model_config.model_id)
        ]
        hf_model_config.model_max_length = model_lengths[0] if model_lengths else -1

        num_params = [
            num_param
            for pattern, num_param in NUM_PARAMS_MAPPING.items()
            if re.match(pattern=pattern, string=model_config.model_id)
        ]
        hf_model_config.num_params = num_params[0] if num_params else -1

        hf_model_config.id2label = dataset_config.id2label
        hf_model_config.label2id = dataset_config.label2id
        hf_model_config.eos_token_id = hf_model_config.vocab_size - 1
        hf_model_config.bos_token_id = hf_model_config.vocab_size - 1
        hf_model_config.pad_token_id = hf_model_config.vocab_size - 1

        # Check if the vocab size is correct, and if not then correct it
        tok = OpenAITokenizer(
            model_config=model_config, hf_model_config=hf_model_config
        )
        for idx in range(hf_model_config.vocab_size - 1, 0, -1):
            try:
                tok.decode([idx])
                hf_model_config.vocab_size = idx + 1
                break
            except Exception:
                pass
        else:
            raise ValueError(
                f"Couldn't find vocab size for the model {model_config.model_id!r}"
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

        # If the model is a chat model then we need to reduce the maximum context
        # length by 7 tokens, as these are used in the chat prompt
        if model.is_chat_model:
            hf_model_config.model_max_length -= 7
            tokenizer.hf_model_config = hf_model_config
            model.config = hf_model_config

        return model, tokenizer
