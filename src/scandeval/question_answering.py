"""Question-answering benchmark dataset."""

import logging
from functools import partial
from typing import Callable, List, Optional

from datasets.arrow_dataset import Dataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from scandeval.utils import get_special_token_metadata

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark
from .question_answering_trainer import QuestionAnsweringTrainer

# Set up logger
logger = logging.getLogger(__name__)


class QuestionAnswering(BenchmarkDataset):
    """Question-answering benchmark dataset.

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
            Hugging Face dataset: The preprocessed dataset.
        """
        split: str = kwargs.pop("split")
        tokenizer: PreTrainedTokenizer = kwargs.pop("tokenizer")

        # Choose the preprocessing function depending on the dataset split
        if split == "test":
            preprocess_fn = partial(prepare_test_examples, tokenizer=tokenizer)
        else:
            preprocess_fn = partial(prepare_train_examples, tokenizer=tokenizer)

        # Preprocess the data and return it
        try:
            preprocessed = dataset.map(
                preprocess_fn,
                batched=True,
                batch_size=10,
                remove_columns=dataset.column_names,
            )
        except NotImplementedError as e:
            raise InvalidBenchmark(str(e))

        # The Trainer hides the columns that are not used by the model (here `id` and
        # `offset_mapping` which we will need for our post-processing), so we set them
        # back
        preprocessed.set_format(
            type=preprocessed.format["type"],
            columns=list(preprocessed.features.keys()),
        )

        # Return the preprocessed dataset
        return preprocessed

    def _get_trainer(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_collator: DataCollator,
        compute_metrics: Callable,
        callbacks: List[TrainerCallback],
    ) -> Trainer:
        return QuestionAnsweringTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def _evaluate_dataset(
        self,
        trainer: QuestionAnsweringTrainer,
        dataset: Dataset,
        prepared_dataset: Dataset,
        metric_key_prefix: str,
    ):
        return trainer.evaluate(
            orig_eval_dataset=dataset,
            eval_dataset=prepared_dataset,
            metric_key_prefix=metric_key_prefix,
        )

    def _load_data_collator(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Hugging Face tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorWithPadding(tokenizer)


def prepare_train_examples(
    examples: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
) -> BatchEncoding:
    """Prepare the features for training.

    Args:
        examples (BatchEncoding):
            The examples to prepare.

    Returns:
        BatchEncoding:
            The prepared examples.
    """
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Extract special token metadata from the tokenizer
    special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
    has_cls_token = special_token_metadata["has_cls_token"]
    has_sep_token = special_token_metadata["has_sep_token"]
    cls_token_id = special_token_metadata["cls_token_id"]
    cls_token = special_token_metadata["cls_token"]
    sep_token = special_token_metadata["sep_token"]

    # If the tokenizer is not adding special tokens, then we add them manually
    if not has_cls_token and not has_sep_token:
        examples["question"] = [
            f"{cls_token}{q}{sep_token}" for q in examples["question"]
        ]
        examples["context"] = [f"{c}{sep_token}" for c in examples["context"]]

    # Compute the stride, being a quarter of the context length
    stride = tokenizer.model_max_length // 4
    max_length = tokenizer.model_max_length - stride

    # Tokenize our examples with truncation and padding, but keep the overflows using a
    # stride. This results in one example possible giving several features when a
    # context is long, each of those features having a context that overlaps a bit the
    # context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we
    # need a map from a feature to its corresponding example. This key gives us just
    # that
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character position in the
    # original context. This will help us compute the start_positions and
    # end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialise the start- and end positions of the answers
    tokenized_examples["start_positions"] = list()
    tokenized_examples["end_positions"] = list()

    for i, offsets in enumerate(offset_mapping):

        # Get the input IDs for the current example
        input_ids = tokenized_examples.input_ids[i]

        # We will label impossible answers with the index of the CLS token
        cls_index = input_ids.index(cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context
        # and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Manually ensure that the special tokens are set to None in `sequence_ids`
        for special_token in tokenizer.special_tokens_map.keys():
            special_token_id = getattr(tokenizer, f"{special_token}_id")
            if special_token_id is not None:
                sequence_ids = [
                    None if token_id == special_token_id else seq_id
                    for token_id, seq_id in zip(input_ids, sequence_ids)
                ]

        # One example can give several spans, this is the index of the example
        # containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples.start_positions.append(cls_index)
            tokenized_examples.end_positions.append(cls_index)

        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is
            # labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples.start_positions.append(cls_index)
                tokenized_examples.end_positions.append(cls_index)

            # Otherwise move the token_start_index and token_end_index to the two ends
            # of the answer. Note: we could go after the last offset if the answer is
            # the last word (edge case).
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples.start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples.end_positions.append(token_end_index + 1)

    return tokenized_examples


def prepare_test_examples(
    examples: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
) -> BatchEncoding:
    """Prepare test examples.

    Args:
        examples (BatchEncoding):
            Dictionary of test examples.
        tokenizer (Hugging Face tokenizer):
            The tokenizer used to preprocess the examples.

    Returns:
        BatchEncoding:
            The prepared test examples.
    """
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Extract special token metadata from the tokenizer
    special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
    has_cls_token = special_token_metadata["has_cls_token"]
    has_sep_token = special_token_metadata["has_sep_token"]
    cls_token = special_token_metadata["cls_token"]
    sep_token = special_token_metadata["sep_token"]

    # If the tokenizer is not adding special tokens, then we add them manually
    if not has_cls_token and not has_sep_token:
        examples["question"] = [
            f"{cls_token}{q}{sep_token}" for q in examples["question"]
        ]
        examples["context"] = [f"{c}{sep_token}" for c in examples["context"]]

    # Compute the stride, being a quarter of the context length
    stride = tokenizer.model_max_length // 4
    max_length = tokenizer.model_max_length - stride

    # Tokenize our examples with truncation and maybe padding, but keep the overflows
    # using a stride. This results in one example possible giving several features when
    # a context is long, each of those features having a context that overlaps a bit
    # the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we
    # need a map from a feature to its corresponding example. This key gives us just
    # that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the id that gave us this feature and we will store the offset mappings.
    tokenized_examples["id"] = list()

    for i in range(len(tokenized_examples.input_ids)):

        # Grab the sequence corresponding to that example (to know what is the context
        # and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example
        # containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples.id.append(examples["id"][sample_index])

        # Set to (-1, -1) the offset_mapping that are not part of the context so it's
        # easy to determine if a token position is part of the context or not.
        tokenized_examples.offset_mapping[i] = [
            (o if sequence_ids[k] == context_index else (-1, -1))
            for k, o in enumerate(tokenized_examples.offset_mapping[i])
        ]

    return tokenized_examples
