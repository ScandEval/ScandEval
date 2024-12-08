"""Abstract benchmark module class that the model classes inherit from."""

import collections.abc as c
import logging
import sys
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property, partial

from datasets import DatasetDict
from torch import nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, Trainer

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import BatchingPreference
from ..exceptions import NeedsEnvironmentVariable, NeedsExtraInstalled
from ..task_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..types import ComputeMetricsFunction, ExtractLabelsFunction

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
        # Set logging level based on verbosity
        if hasattr(sys, "_called_from_test"):
            logging_level = logging.CRITICAL
        elif self.benchmark_config.verbose:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO
        logger.setLevel(logging_level)

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

    @cached_property
    @abstractmethod
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        ...

    @cached_property
    def compute_metrics(self) -> ComputeMetricsFunction:
        """The function used to compute the metrics.

        Returns:
            The function used to compute the metrics.
        """
        match self.dataset_config.task.supertask:
            case "sequence-classification":
                return partial(
                    sequence_classification.compute_metrics,
                    id2label=self.dataset_config.id2label,
                    dataset_config=self.dataset_config,
                    benchmark_config=self.benchmark_config,
                )
            case "text-to-text":
                return partial(
                    text_to_text.compute_metrics,
                    dataset_config=self.dataset_config,
                    benchmark_config=self.benchmark_config,
                )
            case "token-classification":
                return partial(
                    token_classification.compute_metrics,
                    has_misc_tags=self._has_misc_tags,
                    dataset_config=self.dataset_config,
                    benchmark_config=self.benchmark_config,
                )
            case "question-answering":
                return partial(
                    question_answering.compute_metrics,
                    dataset_config=self.dataset_config,
                    benchmark_config=self.benchmark_config,
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {self.dataset_config.task.supertask}."
                )

    @cached_property
    @abstractmethod
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        ...

    @cached_property
    @abstractmethod
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
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
            if self.dataset_config.task.supertask == "token-classification":
                labels_in_train: set[str] = {
                    tag for tag_list in dataset["train"]["labels"] for tag in tag_list
                }
                self._has_misc_tags = (
                    "B-MISC" in labels_in_train or "I-MISC" in labels_in_train
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
