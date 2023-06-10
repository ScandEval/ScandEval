"""Base class for a model setup."""

from typing import Callable, Protocol

from torch import LongTensor, Tensor
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer
from transformers.utils import ModelOutput

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig


class GenerativeModel(Protocol):
    """A protocol for a generative model."""

    def generate(
        self,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        prefix_allowed_tokens_fn: Callable[[int, Tensor], list[int]] | None = None,
        synced_gpus: bool | None = None,
        assistant_model: PreTrainedModel | None = None,
        streamer: BaseStreamer | None = None,
        **model_kwargs,
    ) -> ModelOutput | LongTensor:
        ...


class ModelSetup(Protocol):
    """A protocol for a general model setup."""

    def __init__(self, benchmark_config: BenchmarkConfig) -> None:
        ...

    def model_exists(self, model_id: str) -> bool:
        ...

    def get_model_config(self, model_id: str) -> ModelConfig:
        ...

    def load_model(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> tuple[PreTrainedTokenizer | None, PreTrainedModel | GenerativeModel]:
        ...
