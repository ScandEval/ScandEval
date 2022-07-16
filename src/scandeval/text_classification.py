"""Text classification benchmark dataset."""

import logging
from functools import partial
from typing import Optional

from datasets import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


class TextClassificationBenchmark(BenchmarkDataset):
    """Text classification benchmark dataset.

    Args:
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.

    Attributes:
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.
    """

    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        """Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset (Hugging Face dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face dataset:
                The preprocessed dataset.
        """
        if framework == "pytorch":
            tokenizer = kwargs["tokenizer"]

            def tokenise(examples: dict) -> dict:
                return tokenizer(examples["text"], truncation=True, padding=True)

            tokenised = dataset.map(tokenise, batched=True, load_from_cache_file=False)

            numericalise = partial(
                self._create_numerical_labels, label2id=kwargs["config"].label2id
            )
            preprocessed = tokenised.map(
                numericalise, batched=True, load_from_cache_file=False
            )

            return preprocessed.remove_columns(["text"])

        elif framework == "spacy":
            raise InvalidBenchmark(
                "Evaluation of text predictions for SpaCy models is not yet "
                "implemented."
            )

    def _create_numerical_labels(self, examples: dict, label2id: dict) -> dict:
        try:
            examples["label"] = [label2id[lbl.upper()] for lbl in examples["label"]]
        except KeyError:
            raise InvalidBenchmark(
                f"One of the labels in the dataset, {examples['label'].upper()}, does "
                f"not occur in the label2id dictionary {label2id}."
            )
        return examples

    def _load_data_collator(self, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Hugging Face tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def _get_spacy_predictions_and_labels(self, model, dataset: Dataset) -> tuple:
        """Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the second
                array contains the true labels.
        """
        raise InvalidBenchmark(
            "Evaluation of text classification tasks for SpaCy models is not possible."
        )
