"""Abstract benchmark module class that the model classes inherit from."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property

from datasets import DatasetDict
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import BatchingPreference
from ..exceptions import NeedsEnvironmentVariable, NeedsExtraInstalled

logger = logging.getLogger("scandeval")


class BenchmarkModule(ABC):
    """Abstract class for a benchmark module.

    Attributes:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.
        is_generative:
            Whether the model is generative.
    """

    _is_generative: bool | None
    batching_preference: BatchingPreference

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the benchmark module.

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
        self._log_metadata()

    def _log_metadata(self) -> None:
        """Log the metadata of the model."""
        logging_msg: str = ""
        if self.num_params < 0:
            logging_msg += "The model has an unknown number of parameters, "
        else:
            logging_msg += f"The model has {self.num_params:,} parameters, "
        if self.vocab_size < 0:
            logging_msg += "an unknown vocabulary size, "
        else:
            logging_msg += f"a vocabulary size of {self.vocab_size:,}, "
        if self.model_max_length < 0:
            logging_msg += "and an unknown maximum sequence length."
        else:
            logging_msg += f"and a maximum context length of {self.model_max_length:,}."
        logger.info(logging_msg)

    def get_pytorch_module(self) -> "nn.Module":
        """Get the underlying PyTorch module.

        Returns:
            The PyTorch module.
        """
        if hasattr(self, "_model"):
            return self._model
        raise NotImplementedError(
            "The `get_pytorch_module` method has not been implemented for "
            f"{self.__class__.__name__}."
        )

    def get_tokenizer(self) -> "PreTrainedTokenizer":
        """Get the underlying tokenizer.

        Returns:
            The tokenizer.
        """
        if hasattr(self, "_tokenizer"):
            return self._tokenizer
        raise NotImplementedError(
            "The `get_tokenizer` method has not been implemented for "
            f"{self.__class__.__name__}."
        )

    @cached_property
    def is_generative(self) -> bool:
        """Whether the model is generative.

        Returns:
            Whether the model is generative.
        """
        if self._is_generative is not None:
            return self._is_generative
        raise NotImplementedError(
            "The model type must define whether it is generative or not."
        )

    @cached_property
    @abstractmethod
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        ...

    @cached_property
    @abstractmethod
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        ...

    @cached_property
    @abstractmethod
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        ...

    def prepare_datasets(
        self, datasets: list[DatasetDict], task: Task
    ) -> list[DatasetDict]:
        """Prepare the datasets for the model.

        This includes things like tokenisation.

        Args:
            datasets:
                The datasets to prepare.
            task:
                The task to prepare the datasets for.

        Returns:
            The prepared datasets.
        """
        for idx, dataset in enumerate(
            tqdm(iterable=datasets, desc="Preparing datasets")
        ):
            prepared_dataset = self.prepare_dataset(
                dataset=dataset, task=task, itr_idx=idx
            )
            datasets[idx] = DatasetDict(
                dict(
                    train=prepared_dataset["train"],
                    val=prepared_dataset["val"],
                    test=prepared_dataset["test"],
                    original_train=dataset["train"],
                    original_val=dataset["val"],
                    original_test=dataset["test"],
                )
            )
        return datasets

    @abstractmethod
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
        ...

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        raise NotImplementedError(
            "The `generate` method has not been implemented for "
            f"{self.__class__.__name__}."
        )

    @classmethod
    @abstractmethod
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
        ...

    @classmethod
    @abstractmethod
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
        ...
