"""Protocols used throughout the project."""

from typing import TYPE_CHECKING, Literal, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    import torch
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

    cls_token: str | None
    sep_token: str | None
    bos_token: str | None
    eos_token: str | None
    pad_token: str | None
    unk_token: str | None
    cls_token_id: int
    sep_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_token_id: int
    is_fast: bool
    chat_template: str | None
    special_tokens_map: dict[str, str]
    model_max_length: int
    vocab_size: int
    padding_side: Literal["right", "left"]

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

    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode one or more texts.

        Args:
            text:
                The text to encode.
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

    def pad(
        self,
        encoded_inputs: Union[
            "BatchEncoding",
            list["BatchEncoding"],
            dict[str, list[int]],
            dict[str, list[list[int]]],
            list[dict[str, list[int]]],
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

    config: "PretrainedConfig"
    device: "torch.device"

    def to(self, device: "torch.device") -> "GenerativeModel":
        """Move the model to a device.

        Args:
            device:
                The device to move the model to.

        Returns:
            The model.
        """
        ...

    def eval(self) -> "GenerativeModel":
        """Put the model in evaluation mode.

        Returns:
            The model.
        """
        ...

    def generate(
        self,
        inputs: "torch.Tensor",
        generation_config: "GenerationConfig | None" = None,
        **generation_kwargs,
    ) -> "ModelOutput | torch.Tensor":
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

    def __call__(
        self, input_ids: "torch.Tensor", labels: "torch.Tensor | None" = None
    ) -> "ModelOutput":
        """Call the model.

        Args:
            input_ids:
                The input IDs.
            labels:
                The labels.

        Returns:
            The model output.
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
