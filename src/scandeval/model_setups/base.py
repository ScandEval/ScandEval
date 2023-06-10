"""Base class for a model setup."""

from typing import Protocol

from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig


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
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        ...
