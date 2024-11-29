"""Model and tokenizer wrapper for Anthopic models."""

import importlib.util
from typing import TYPE_CHECKING

from transformers.modeling_utils import ModelOutput

from ..config import BenchmarkConfig
from ..exceptions import NeedsExtraInstalled

if importlib.util.find_spec("anthropic") is not None:
    from anthropic import Anthropic

if TYPE_CHECKING:
    from torch import LongTensor, Tensor
    from transformers import GenerationConfig


class AnthropicModel:
    """An Anthropic model."""

    def __init__(self, benchmark_config: BenchmarkConfig) -> None:
        """Initialise the model."""
        if importlib.util.find_spec("anthropic") is None:
            raise NeedsExtraInstalled(extra="anthropic")

        self.benchmark_config = benchmark_config
        self.client = Anthropic(api_key=self.benchmark_config.anthropic_api_key)

    def generate(
        self,
        inputs: "Tensor",
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> "Tensor | LongTensor | ModelOutput":
        """Generate text using the model.

        Args:
            inputs:
                The input IDs, of shape (batch_size, sequence_length).
            generation_config:
                The generation configuration. If None then a default GenerationConfig
                will be used. Defaults to None.
            **generation_kwargs:
                Additional keyword arguments. Can also be used to override
                generation configuration.

        Returns:
            The model output.
        """
        raise NotImplementedError
