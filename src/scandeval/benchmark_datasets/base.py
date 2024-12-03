"""Abstract benchmarking dataset class."""

import logging
import sys
import typing as t
from abc import ABC, abstractmethod

import evaluate
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from huggingface_hub.utils import HfHubHTTPError
from numpy.random import Generator
from transformers import DataCollator, DataCollatorWithPadding, Trainer

from ..benchmark_modules import BenchmarkModule
from ..data_models import GenerativeModelOutput
from ..exceptions import InvalidBenchmark
from ..utils import unscramble

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig
    from ..types import Labels, Predictions


logger = logging.getLogger("scandeval")


class BenchmarkDataset(ABC):
    """Abstract benchmarking dataset class.

    Args:
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.

    Attributes:
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
    """

    trainer_class: t.Type[Trainer]

    def __init__(
        self, dataset_config: "DatasetConfig", benchmark_config: "BenchmarkConfig"
    ) -> None:
        """Initialise the dataset.

        Args:
            dataset_config:
                The configuration for the dataset.
            benchmark_config:
                The configuration for the benchmark.
        """
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self._metrics = {
            metric_cfg.name: (
                evaluate.load(
                    path=metric_cfg.huggingface_id,
                    cache_dir=self.benchmark_config.cache_dir,
                )
                if metric_cfg.huggingface_id != ""
                else None
            )
            for metric_cfg in dataset_config.task.metrics
        }

        # Set logging level based on verbosity
        if hasattr(sys, "_called_from_test"):
            logging_level = logging.CRITICAL
        elif self.benchmark_config.verbose:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO
        logger.setLevel(logging_level)

    def load_data(self, rng: Generator) -> list[DatasetDict]:
        """Load the raw bootstrapped datasets.

        Args:
            rng:
                The random number generator to use.

        Returns:
            A list of bootstrapped datasets, one for each iteration.
        """
        try:
            dataset = load_dataset(
                path=self.dataset_config.huggingface_id,
                cache_dir=self.benchmark_config.cache_dir,
                token=unscramble("HjccJFhIozVymqXDVqTUTXKvYhZMTbfIjMxG_"),
            )
        except HfHubHTTPError:
            raise InvalidBenchmark("The Hugging Face Hub seems to be down.")

        assert isinstance(dataset, DatasetDict)

        dataset = DatasetDict({key: dataset[key] for key in ["train", "val", "test"]})
        dataset = self._process_data(dataset_dict=dataset)

        if self.benchmark_config.only_validation_split:
            dataset["test"] = dataset["val"]

        # Remove empty examples from the datasets
        for text_feature in ["tokens", "text"]:
            if text_feature in dataset["train"].features:
                dataset = dataset.filter(lambda x: len(x[text_feature]) > 0)

        # If we are testing then truncate the test set
        if hasattr(sys, "_called_from_test"):
            dataset["test"] = dataset["test"].select(range(1))

        # Bootstrap the splits
        bootstrapped_splits: dict[str, list[Dataset]] = dict()
        for split in ["train", "val", "test"]:
            bootstrap_indices = rng.integers(
                0,
                len(dataset[split]),
                size=(self.benchmark_config.num_iterations, len(dataset[split])),
            )
            bootstrapped_splits[split] = [
                dataset[split].select(bootstrap_indices[idx])
                for idx in range(self.benchmark_config.num_iterations)
            ]

        datasets = [
            DatasetDict(
                {
                    split: bootstrapped_splits[split][idx]
                    for split in ["train", "val", "test"]
                }
            )
            for idx in range(self.benchmark_config.num_iterations)
        ]
        return datasets

    def _process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Process the data.

        Args:
            dataset_dict:
                The dataset dictionary.

        Returns:
            The processed dataset dictionary.
        """
        return dataset_dict

    def get_data_collator(self, model: "BenchmarkModule") -> "DataCollator":
        """Load the data collator used to prepare samples during finetuning.

        Args:
            model:
                The model to use with the data collator.

        Returns:
            The data collator.
        """
        assert hasattr(model, "_tokenizer"), (
            "The tokenizer must be set in the model in order to load the data "
            "collator."
        )
        return DataCollatorWithPadding(model._tokenizer, padding="longest")

    @abstractmethod
    def compute_metrics(
        self,
        model_outputs_and_labels: tuple["Predictions", "Labels"],
        id2label: dict[int, str],
    ) -> dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            model_outputs_and_labels:
                The first sequence contains the model outputs and the second sequence
                contains the true labels.
            id2label:
                Conversion of indices to labels.

        Returns:
            A dictionary with the names of the metrics as keys and the metric values as
            values.
        """
        pass

    @abstractmethod
    def extract_labels_from_generation(
        self, input_batch: dict[str, list], model_output: "GenerativeModelOutput"
    ) -> list[t.Any]:
        """Extract the predicted labels from the generated output.

        Args:
            input_batch:
                The input batch, where the keys are the feature names and the values
                are lists with the feature values.
            model_output:
                The raw generated output of the model.

        Returns:
            The predicted labels.
        """
        pass
