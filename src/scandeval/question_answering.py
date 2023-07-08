"""Question-answering benchmark dataset."""

import logging
from functools import partial
from typing import Any, Callable

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.modeling_utils import ModelOutput, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark
from .generation import extract_raw_predictions
from .model_setups import GenerativeModel, Tokenizer
from .question_answering_trainer import QuestionAnsweringTrainer
from .utils import get_special_token_metadata

# Set up logger
logger = logging.getLogger(__package__)


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
        generative_model: bool = kwargs.pop("generative_model")

        # If the tokenizer is not a fast variant then raise an error
        if not tokenizer.is_fast and not generative_model:
            raise InvalidBenchmark(
                "Question-answering benchmarks require a fast tokenizer."
            )

        if generative_model:
            preprocess_fn = partial(
                prepare_examples_for_generation,
                tokenizer=tokenizer,
            )
        elif split == "test":
            preprocess_fn = partial(prepare_test_examples, tokenizer=tokenizer)
        else:
            preprocess_fn = partial(prepare_train_examples, tokenizer=tokenizer)

        # Preprocess the data and return it
        try:
            preprocessed = dataset.map(
                preprocess_fn,
                batched=True,
                batch_size=10,
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
        model: PreTrainedModel | GenerativeModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: Tokenizer,
        compute_metrics: Callable,
        callbacks: list[TrainerCallback],
    ) -> Trainer:
        return QuestionAnsweringTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
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

    def _load_data_collator(
        self,
        tokenizer: Tokenizer | None = None,
        model: PreTrainedModel | GenerativeModel | None = None,
    ):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.
            model (PreTrainedModel or GenerativeModel or None, optional):
                A pretrained model. Can be None if the model is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorWithPadding(tokenizer=tokenizer)

    def _compute_metrics(
        self,
        model_outputs_and_labels: tuple[list, list],
        id2label: list[str],
    ) -> dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            model_outputs_and_labels (pair of sequences):
                The first sequence contains the model outputs and the second sequence
                contains the true labels.
            id2label (list of str):
                Conversion of indices to labels.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        model_outputs, labels = model_outputs_and_labels

        model_output_dtype = np.asarray(model_outputs).dtype
        if model_output_dtype in [np.float16, np.float32, np.float64]:
            predictions = np.asarray(model_outputs).argmax(axis=-1)
        else:
            predictions = model_outputs

        results: dict[str, float] = dict()
        for cfg in self.dataset_config.task.metrics:
            metric = self._metrics[cfg.name]
            score_dict: dict[str, float] | None = metric.compute(
                predictions=predictions,
                references=labels,
                **cfg.compute_kwargs,
            )
            if score_dict is not None:
                scores = score_dict[cfg.results_key]
                results[cfg.name] = scores
        return results

    def _extract_few_shot_examples(
        self, train_dataset: Dataset, random_seed: int
    ) -> list[dict[str, Any]]:
        """Extract few-shot examples from the training dataset.

        Args:
            train_dataset (Hugging Face dataset):
                The training dataset.
            random_seed (int):
                The random seed to use when extracting the few-shot examples.

        Returns:
            list[dict[str, Any]]:
                The few-shot examples.
        """
        train_with_short_examples = train_dataset.filter(
            lambda example: len(example["context"]) < 512
        )
        shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
        num_few_shots = self.dataset_config.num_few_shot_examples
        few_shot_examples: list[dict[str, Any]] = [
            example for example in shuffled_train.select(range(num_few_shots))
        ]
        return few_shot_examples

    def _apply_few_shot_prompt(
        self, examples: dict, few_shot_examples: list[dict]
    ) -> dict:
        """Apply a few-shot prompt to the examples.

        Args:
            examples (dict):
                The examples to apply the prompt to.
            few_shot_examples (list of dict):
                The examples to be included in the few-shot prompt.

        Returns:
            dict:
                The examples with the few-shot prompt applied.
        """
        # Build the few-shot part of the prompt
        few_shot_prompts = [
            self.dataset_config.prompt_template.format(
                text=example["context"].replace("\n", " ").strip(),
                question=example["question"].strip(),
                label=example["answers"]["text"][0],
            )
            for example in few_shot_examples
        ]
        prompt_prefix = ""
        if self.dataset_config.prompt_prefix:
            prompt_prefix = self.dataset_config.prompt_prefix + "\n\n"
        few_shot_prompt = prompt_prefix + "\n\n".join(few_shot_prompts)

        # Add the texts from the examples to the prompts
        new_prompts = [
            self.dataset_config.prompt_template.format(
                text=context.replace("\n", " ").strip(), question=question, label=""
            )
            for context, question in zip(examples["context"], examples["question"])
        ]
        examples["text"] = [
            few_shot_prompt + "\n\n" + new_prompt for new_prompt in new_prompts
        ]

        return examples

    def _extract_labels_from_generation(
        self,
        input_batch: dict[str, list],
        model_output: ModelOutput,
        tokenizer: Tokenizer,
    ) -> list[Any]:
        """Extract the predicted labels from the generated output.

        Args:
            input_batch (dict):
                The input batch, where the keys are the feature names and the values
                are lists with the feature values.
            model_output (ModelOutput):
                The raw generated output of the model.
            tokenizer (Tokenizer):
                The tokenizer used together with the model.

        Returns:
            list:
                The predicted labels.
        """
        raw_predictions = extract_raw_predictions(
            generated_sequences=model_output["sequences"],
            tokenizer=tokenizer,
            dataset_config=self.dataset_config,
        )

        predictions = [
            dict(
                id=id,
                prediction_text=predicted_answer.lower(),
                no_answer_probability=0.0,
            )
            for id, predicted_answer in zip(input_batch["id"], raw_predictions)
        ]

        return predictions


def prepare_train_examples(
    examples: BatchEncoding,
    tokenizer: Tokenizer,
) -> BatchEncoding:
    """Prepare the features for training.

    Args:
        examples (BatchEncoding):
            The examples to prepare.
        tokenizer (Tokenizer):
            The tokenizer to use to prepare the examples.

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
        text_pair=examples["context"],
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
            if hasattr(tokenizer, f"{special_token}_id"):
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
    tokenizer: Tokenizer,
) -> BatchEncoding:
    """Prepare test examples.

    Args:
        examples (BatchEncoding):
            Dictionary of test examples.
        tokenizer (Tokenizer):
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
        text_pair=examples["context"],
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


def prepare_examples_for_generation(
    examples: BatchEncoding, tokenizer: Tokenizer
) -> BatchEncoding:
    """Prepare test examples.

    Args:
        examples (BatchEncoding):
            Dictionary of test examples.
        tokenizer (Tokenizer):
            The tokenizer used to preprocess the examples.

    Returns:
        BatchEncoding:
            The prepared test examples.
    """
    tokenized_examples = tokenizer(text=examples["text"], truncation=True)
    tokenized_examples["label"] = [
        dict(
            id=id,
            answers=dict(
                answer_start=answer_dct["answer_start"],
                text=[answer_text.lower() for answer_text in answer_dct["text"]],
            ),
        )
        for id, answer_dct in zip(examples["id"], examples["answers"])
    ]
    return tokenized_examples
