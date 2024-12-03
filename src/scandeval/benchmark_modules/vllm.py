"""Generative models using the vLLM inference framework."""

import importlib.util
import typing as t
from functools import cached_property

from datasets import DatasetDict
from transformers import PreTrainedTokenizer

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..exceptions import NeedsEnvironmentVariable, NeedsExtraInstalled
from .base import BenchmarkModule

if t.TYPE_CHECKING or importlib.util.find_spec("vllm") is not None:
    from vllm import LLM


class VLLMModel(BenchmarkModule):
    """A generative model using the vLLM inference framework."""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self._model, self._tokenizer = self._load_model_and_tokenizer()
        self._log_metadata()

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        raise NotImplementedError

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        raise NotImplementedError

    @cached_property
    def model_max_length(self) -> int:
        """The maximum context length of the model.

        Returns:
            The maximum context length of the model.
        """
        raise NotImplementedError

    def prepare_dataset(
        self, dataset: DatasetDict, task: Task, itr_idx: int
    ) -> DatasetDict:
        """Prepare the dataset for the model.

        This includes things like tokenisation.

        Args:
            dataset:
                The dataset to prepare.
            task:
                The task to prepare the dataset for.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The prepared dataset.
        """
        raise NotImplementedError

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        raise NotImplementedError

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        return False  # TODO: Implement thisj

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        raise NotImplementedError

    def _load_model_and_tokenizer(self) -> "tuple[LLM, PreTrainedTokenizer]":
        """Load the model and tokenizer.

        Returns:
            The loaded model and tokenizer.
        """
        raise NotImplementedError
