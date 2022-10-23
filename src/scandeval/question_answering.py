"""Question-answering benchmark dataset."""

import logging
from collections import defaultdict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.data.data_collator import DefaultDataCollator
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from scandeval.protocols import DataCollator, Model, TokenizedOutputs, Tokenizer
from scandeval.question_answering_trainer import QuestionAnsweringTrainer

from .benchmark_dataset import BenchmarkDataset

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
        tokenizer: Tokenizer = kwargs.pop("tokenizer")

        # Choose the preprocessing function depending on the dataset split
        if split == "train":
            preprocess_fn = partial(
                prepare_train_examples,
                tokenizer=tokenizer,
            )
        else:
            preprocess_fn = partial(
                prepare_test_examples,
                tokenizer=tokenizer,
            )

        # Preprocess the data and return it
        preprocessed = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return preprocessed

    def _get_trainer(
        self,
        model: Model,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: Tokenizer,
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
            post_process_function=postprocess_predictions,
        )

    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:

        # Extract the predictions and labels
        predictions = postprocess_predictions(
            predictions=predictions,
            dataset=dataset,
            prepared_dataset=prepared_dataset,
            cls_token_index=kwargs["cls_token_index"],
        )
        labels = postprocess_labels(dataset=dataset)

        # Package the predictions and labels into the standard format and return them
        return [(predictions, labels)]

    def _load_data_collator(self, tokenizer: Optional[Tokenizer] = None):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Hugging Face tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DefaultDataCollator(tokenizer)


def prepare_train_examples(
    examples: BatchEncoding,
    tokenizer: Tokenizer,
) -> TokenizedOutputs:
    """Prepare the features for training.

    Args:
        examples (BatchEncoding):
            The examples to prepare.

    Returns:
        TokenizedOutputs:
            The prepared examples.
    """

    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

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

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):

        # We will label impossible answers with the index of the CLS token
        input_ids = tokenized_examples.input_ids[i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context
        # and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

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
    tokenizer: Tokenizer,
) -> TokenizedOutputs:
    """Prepare test examples.

    Args:
        examples (BatchEncoding):
            Dictionary of test examples.
        tokenizer (Hugging Face tokenizer):
            The tokenizer used to preprocess the examples.

    Returns:
        TokenizedOutputs:
            The prepared test examples.
    """
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

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
    tokenized_examples.id = list()

    for i in range(len(tokenized_examples.input_ids)):

        # Grab the sequence corresponding to that example (to know what is the context
        # and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example
        # containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples.id.append(examples.id[sample_index])

        # Set to (-1, -1) the offset_mapping that are not part of the context so it's
        # easy to determine if a token position is part of the context or not.
        tokenized_examples.offset_mapping[i] = [
            (o if sequence_ids[k] == context_index else (-1, -1))
            for k, o in enumerate(tokenized_examples.offset_mapping[i])
        ]

    return tokenized_examples


def postprocess_predictions(
    predictions: Sequence,
    dataset: Dataset,
    prepared_dataset: Dataset,
    cls_token_index: int,
) -> List[dict]:
    """Postprocess the predictions, to allow easier metric computation.

    Args:
        predictions (Sequence):
            The predictions to postprocess.
        dataset (Dataset):
            The dataset containing the examples.
        prepared_dataset (Dataset):
            The dataset containing the prepared examples.
        cls_token_index (int):
            The index of the CLS token.

    Returns:
        list of dicts:
            The postprocessed predictions.
    """
    # Extract the logits from the predictions
    all_start_logits = np.asarray(predictions)[:, :, 0]
    all_end_logits = np.asarray(predictions)[:, :, 1]

    # Build a map from an example to its corresponding features, being the blocks of
    # text from the context that we're feeding into the model. An example can have
    # multiple features/blocks if it has a long context.
    id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(prepared_dataset):
        id = feature["id"]
        example_index = id_to_index[id]
        features_per_example[example_index].append(i)

    # Loop over all the examples
    predictions = list()
    for example_index, example in enumerate(dataset):

        # Extract the best valid answer associated with the current example
        best_answer = find_best_answer(
            all_start_logits=all_start_logits,
            all_end_logits=all_end_logits,
            prepared_dataset=prepared_dataset,
            feature_indices=features_per_example[example_index],
            context=example["context"],
            max_answer_length=30,
            num_best_logits=20,
            min_null_score=0.0,
            cls_token_index=cls_token_index,
        )

        # Create the final prediction dictionary, to be added to the list of
        # predictions
        prediction = dict(
            id=example["id"],
            prediction_text=best_answer,
            no_answer_probability=0.0,
        )

        # Add the answer to the list of predictions
        predictions.append(prediction)

    return predictions


def find_best_answer(
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    prepared_dataset: Dataset,
    feature_indices: List[int],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
    cls_token_index: int,
) -> str:
    """Find the best answer for a given example.

    Args:
        all_start_logits (NumPy array):
            The start logits for all the features.
        all_end_logits (NumPy array):
            The end logits for all the features.
        prepared_dataset (Dataset):
            The dataset containing the prepared examples.
        feature_indices (list of int):
            The indices of the features associated with the current example.
        context (str):
            The context of the example.
        max_answer_length (int):
            The maximum length of the answer.
        num_best_logits (int):
            The number of best logits to consider.
        min_null_score (float):
            The minimum score an answer can have.
        cls_token_index (int):
            The index of the CLS token.

    Returns:
        str:
            The best answer for the example.
    """
    # Loop through all the features associated to the current example
    valid_answers = list()
    for feature_index in feature_indices:

        # Get the features associated with the current example
        features = prepared_dataset[feature_index]

        # Get the predictions of the model for this feature
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]

        # Update minimum null prediction
        cls_index = features["input_ids"].index(cls_token_index)
        feature_null_score = (start_logits[cls_index] + end_logits[cls_index]).item()
        if min_null_score < feature_null_score:
            min_null_score = feature_null_score

        # Find the valid answers for the feature
        valid_answers_for_feature = find_valid_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=features["offset_mapping"],
            context=context,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        valid_answers.extend(valid_answers_for_feature)

    # In the very rare edge case we have not a single non-null prediction, we create a
    # fake prediction to avoid failure
    if not valid_answers:
        return ""

    # Otherwise, we select the answer with the largest score as the best answer, and
    # return it
    best_answer_dict = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    return best_answer_dict["text"]


def find_valid_answers(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: List[Tuple[int, int]],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
) -> List[dict]:
    """Find the valid answers from the start and end indexes.

    Args:
        start_logits (NumPy array):
            The logits for the start of the answer.
        end_logits (NumPy array):
            The logits for the end of the answer.
        offset_mapping (list of pairs of int):
            The offset mapping, being a list of pairs of integers for each token index,
            containing the start and end character index in the original context.
        max_answer_length (int):
            The maximum length of the answer.
        num_best_logits (int):
            The number of best logits to consider. Note that this function will run in
            O(`num_best_logits` ^ 2) time.
        min_null_score (float):
            The minimum score an answer can have.

    Returns:
        list of dicts:
            A list of the valid answers, each being a dictionary with keys "text" and
            "score", the score being the sum of the start and end logits.
    """
    # Fetch the top-k predictions for the start- and end token indices
    start_indexes = np.argsort(start_logits)[-1 : -num_best_logits - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -num_best_logits - 1 : -1].tolist()

    # We loop over all combinations of starting and ending indexes for valid answers
    valid_answers = list()
    for start_index in start_indexes:
        for end_index in end_indexes:

            # If the starting or ending index is out-of-scope, meaning that they are
            # either out of bounds or correspond to part of the input_ids that are not
            # in the context, then we skip this index
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or tuple(offset_mapping[start_index]) == (-1, -1)
                or tuple(offset_mapping[end_index]) == (-1, -1)
            ):
                continue

            # Do not consider answers with a length that is either negative or greater
            # than the context length
            max_val = max_answer_length + start_index - 1
            if end_index < start_index or end_index > max_val:
                continue

            # If we got to this point then the answer is valid, so we store the
            # corresponding start- and end character indices in the original context,
            # and from these extract the answer
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            text = context[start_char:end_char]

            # Compute the score of the answer, being the sum of the start and end
            # logits. Intuitively, this indicates how likely the answer is to be
            # correct, and allows us to pick the best valid answer.
            score = start_logits[start_index] + end_logits[end_index]

            # Add the answer to the list of valid answers, if the score is greater
            # than the minimum null score
            if score > min_null_score:
                valid_answers.append(dict(score=score, text=text))

    return valid_answers


def postprocess_labels(dataset: Dataset) -> List[dict]:
    """Postprocess the labels, to allow easier metric computation.

    Args:
        dataset (Dataset):
            The dataset containing the examples.

    Returns:
        list of dicts:
             The postprocessed labels.
    """
    labels = list()
    for example in dataset:

        # Create the associated reference dictionary, to be added to the list of
        # references
        label = dict(
            id=example["id"],
            answers=dict(
                text=example["answers"]["text"],
                answer_start=example["answers"]["answer_start"],
            ),
        )

        # Add the answer and label to the list of predictions and labels, respectively
        labels.append(label)

    return labels
