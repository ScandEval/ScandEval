"""Protocols used throughout the project."""

from typing import TYPE_CHECKING, Literal, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor, device
    from transformers import (
        BatchEncoding,
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.utils import ModelOutput

    from .config import BenchmarkConfig, DatasetConfig, ModelConfig


@runtime_checkable
class Tokenizer(Protocol):
    """A protocol for a tokenizer."""

    cls_token: str
    sep_token: str
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str
    cls_token_id: int
    sep_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_token_id: int
    is_fast: bool
    chat_template: str | None

    def __call__(self, text: str | list[str], **kwargs) -> "BatchEncoding":
        """Call the tokenizer.

        Args:
            text:
                The text to tokenize.
            **kwargs:
                Keyword arguments to pass to the tokenizer.

        Returns:
            The encoded inputs.
        """
        ...

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode a list of token IDs.

        Args:
            token_ids:
                The token IDs to decode.
            **kwargs:
                Keyword arguments to pass to the tokenizer.

        Returns:
            The decoded string.
        """
        ...

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        """Decode a batch of token IDs.

        Args:
            sequences:
                The token IDs to decode.
            **kwargs:
                Keyword arguments to pass to the tokenizer.

        Returns:
            The decoded strings.
        """
        ...

    def encode(self, text: str | list[str] | list[int], **kwargs) -> list[int]:
        """Encode one or more texts.

        Args:
            text:
                The text(s) to encode.
            **kwargs:
                Keyword arguments to pass to the tokenizer.

        Returns:
            The encoded token IDs.
        """
        ...

    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = False
    ) -> str | list[str]:
        """Convert a list of token IDs to tokens.

        Args:
            ids:
                The token IDs to convert.
            skip_special_tokens:
                Whether to skip special tokens.

        Returns:
            The tokens.
        """
        ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """Convert a list of tokens to token IDs.

        Args:
            tokens:
                The tokens to convert.

        Returns:
            The token IDs.
        """
        ...

    @property
    def special_tokens_map(self) -> dict[str, str | list[str]]:
        """The mapping from special tokens to their token strings."""
        ...

    @property
    def model_max_length(self) -> int:
        """The maximum length of a sequence that can be processed by the model."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        ...

    def pad(
        self,
        encoded_inputs: Union[
            "BatchEncoding",
            list["BatchEncoding"],
            dict[str, list[str]],
            dict[str, list[list[str]]],
            list[dict[str, list[str]]],
        ],
        **kwargs,
    ) -> "BatchEncoding":
        """Pad a batch of encoded inputs.

        Args:
            encoded_inputs:
                The encoded inputs to pad.
            **kwargs:
                Keyword arguments to pass to the tokenizer.

        Returns:
            The padded encoded inputs.
        """
        ...

    def apply_chat_template(
        self, conversation: list[dict[Literal["role", "content"], str]], **kwargs
    ) -> str | list[int]:
        """Apply a chat template to a conversation.

        Args:
            conversation:
                The conversation.
            **kwargs:
                Keyword arguments to pass to the tokenizer.

        Returns:
            The conversation as a string.
        """
        ...


@runtime_checkable
class GenerativeModel(Protocol):
    """A protocol for a generative model."""

    @property
    def config(self) -> "PretrainedConfig":
        """The Hugging Face model configuration."""
        ...

    @property
    def device(self) -> "device":
        """The device on which the model is running."""
        ...

    def generate(
        self,
        inputs: "Tensor",
        generation_config: "GenerationConfig | None" = None,
        **generation_kwargs,
    ) -> "ModelOutput | Tensor":
        """Generate text.

        Args:
            inputs:
                The input IDs.
            generation_config:
                The generation configuration.
            **generation_kwargs:
                Keyword arguments to pass to the generation method.

        Returns:
            The generated text.
        """
        ...


class ModelSetup(Protocol):
    """A protocol for a general model setup."""

    def __init__(self, benchmark_config: "BenchmarkConfig") -> None:
        """Initialize the model setup.

        Args:
            benchmark_config:
                The benchmark configuration.
        """
        ...

    def model_exists(self, model_id: str) -> bool | dict[str, str]:
        """Check whether a model exists.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exist, or a dictionary explaining why we cannot check
            whether the model exists.
        """
        ...

    def get_model_config(self, model_id: str) -> "ModelConfig":
        """Get the model configuration.

        Args:
            model_id:
                The model ID.

        Returns:
            The model configuration.
        """
        ...

    def load_model(
        self, model_config: "ModelConfig", dataset_config: "DatasetConfig"
    ) -> tuple["PreTrainedModel | GenerativeModel", "Tokenizer"]:
        """Load a model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.

        Returns:
            The tokenizer and model.
        """
        ...
