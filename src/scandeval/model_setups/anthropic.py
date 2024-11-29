"""Model setup for Anthropic models."""

import importlib.util
import logging
import re
from typing import TYPE_CHECKING

from transformers import PretrainedConfig

from scandeval.model_wrappers.anthropic_models import AnthropicModel

from ..config import ModelConfig
from ..enums import Framework, ModelType
from ..model_wrappers import OpenAITokenizer
from ..utils import create_model_cache_dir

if importlib.util.find_spec("anthropic") is not None:
    pass

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..config import BenchmarkConfig, DatasetConfig
    from ..protocols import GenerativeModel, Tokenizer


logger = logging.getLogger(__package__)


# This is a list of the major models that Anthropic has released
CACHED_ANTHROPIC_MODEL_IDS: list[str] = [
    r"claude-[0-9]+(-[0-9]+)*-(opus|sonnet|haiku)-[0-9]{8}"
]


VOCAB_SIZE_MAPPING = {r"claude-[0-9]+(-[0-9]+)*-(opus|sonnet|haiku)-[0-9]{8}": -1}


MODEL_MAX_LENGTH_MAPPING = {
    r"claude-[0-9]+(-[0-9]+)*-(opus|sonnet|haiku)-[0-9]{8}": 200_000
}


NUM_PARAMS_MAPPING = {r"claude-[0-9]+(-[0-9]+)*-(opus|sonnet|haiku)-[0-9]{8}": -1}


class AnthropicModelSetup:
    """Model setup for Anthropic models.

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
        """Check if a model ID denotes an Anthropic model.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exists, or a dictionary explaining why we cannot check
            whether the model exists.
        """
        return any(
            [
                re.match(pattern=model_pattern, string=model_id) is not None
                for model_pattern in CACHED_ANTHROPIC_MODEL_IDS
            ]
        )

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
            model_type=ModelType.ANTHROPIC,
            model_cache_dir=create_model_cache_dir(
                cache_dir=self.benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=None,
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
        hf_model_config.vocab_size = vocab_sizes[0] if vocab_sizes else -1

        model_lengths = [
            model_length
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
        model = AnthropicModel(benchmark_config=self.benchmark_config)

        return model, tokenizer
