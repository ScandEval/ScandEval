"""Base class for a model setup."""

from typing import Protocol, runtime_checkable

import torch
from transformers import (
    BatchEncoding,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig


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

    def __call__(self, text: str | list[str], **kwargs) -> BatchEncoding:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...

    def encode(self, text: str | list[str] | list[int], **kwargs) -> list[int]:
        ...

    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = False
    ) -> str | list[str]:
        ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        ...

    @property
    def special_tokens_map(self) -> dict[str, str | list[str]]:
        ...

    @property
    def model_max_length(self) -> int:
        ...

    def pad(
        self,
        encoded_inputs: BatchEncoding
        | list[BatchEncoding]
        | dict[str, list[str]]
        | dict[str, list[list[str]]]
        | list[dict[str, list[str]]],
        **kwargs,
    ) -> BatchEncoding:
        ...


@runtime_checkable
class GenerativeModel(Protocol):
    """A protocol for a generative model."""

    @property
    def config(self) -> PretrainedConfig:
        ...

    @property
    def device(self) -> torch.device:
        ...

    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> torch.LongTensor:
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
    ) -> tuple[Tokenizer, PreTrainedModel | GenerativeModel]:
        ...
