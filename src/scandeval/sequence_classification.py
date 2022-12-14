"""Sequence classification benchmark dataset."""

import logging
from functools import partial
from typing import Optional

from datasets.arrow_dataset import Dataset
from transformers import BatchEncoding
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding
from transformers.tokenization_utils import PreTrainedTokenizer

from scandeval.utils import get_special_token_metadata

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


class SequenceClassification(BenchmarkDataset):
    """Sequence classification benchmark dataset.

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

    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
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
        tokenizer: PreTrainedTokenizer = kwargs["tokenizer"]

        # Extract special token metadata from the tokenizer
        special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
        has_cls_token = special_token_metadata["has_cls_token"]
        has_sep_token = special_token_metadata["has_sep_token"]
        cls_token = special_token_metadata["cls_token"]
        sep_token = special_token_metadata["sep_token"]

        def tokenise(examples: dict) -> BatchEncoding:

            # If the tokenizer is not adding special tokens, then we add them manually
            if (
                not has_cls_token
                and not has_sep_token
                and cls_token is not None
                and sep_token is not None
            ):
                examples["text"] = [
                    f"{cls_token}{doc}{sep_token}" for doc in examples["text"]
                ]

            return tokenizer(examples["text"], truncation=True, padding=False)

        tokenised = dataset.map(tokenise, batched=True, load_from_cache_file=False)

        numericalise = partial(
            self._create_numerical_labels, label2id=kwargs["config"].label2id
        )
        preprocessed = tokenised.map(
            numericalise, batched=True, load_from_cache_file=False
        )

        return preprocessed.remove_columns(["text"])

    def _create_numerical_labels(self, examples: dict, label2id: dict) -> dict:
        try:
            examples["label"] = [label2id[lbl.upper()] for lbl in examples["label"]]
        except KeyError:
            raise InvalidBenchmark(
                f"One of the labels in the dataset, {examples['label'].upper()}, does "
                f"not occur in the label2id dictionary {label2id}."
            )
        return examples

    def _load_data_collator(
        self, tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> DataCollator:
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
