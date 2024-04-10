"""Text-to-text generation benchmark dataset."""

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np
from transformers.data.data_collator import DataCollatorWithPadding

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark
from .generation import extract_raw_predictions
from .utils import (
    METRIC_ATTRIBUTES_TAKING_UP_MEMORY,
    HiddenPrints,
    clear_memory,
    raise_if_model_output_contains_nan_values,
)

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers import BatchEncoding, PreTrainedModel
    from transformers.utils import ModelOutput

    from .protocols import GenerativeModel, Tokenizer
    from .types import Labels, Predictions


logger = logging.getLogger(__package__)


class TextToText(BenchmarkDataset):
    """Text-to-text benchmark dataset.

    Args:
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Attributes:
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
    """

    def _preprocess_data(self, dataset: "Dataset", **kwargs) -> "Dataset":
        """Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset:
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            The preprocessed dataset.
        """
        tokenizer: "Tokenizer" = kwargs["tokenizer"]

        def tokenise(examples: dict) -> "BatchEncoding":
            return tokenizer(text=examples["text"], truncation=True, padding=False)

        tokenised = dataset.map(
            tokenise, batched=True, load_from_cache_file=False, keep_in_memory=True
        )

        return tokenised

    def _load_data_collator(
        self,
        tokenizer: "Tokenizer | None" = None,
        model: "PreTrainedModel | GenerativeModel | None" = None,
    ):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer:
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.
            model:
                A pretrained model. Can be None if the model is not used. Defaults to
                None.

        Returns:
            The data collator.
        """
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def _compute_metrics(
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
        model_outputs, labels = model_outputs_and_labels

        raise_if_model_output_contains_nan_values(model_output=model_outputs)

        model_output_dtype = np.asarray(model_outputs).dtype
        output_is_prob = model_output_dtype in [np.float16, np.float32, np.float64]
        if output_is_prob:
            predictions = np.asarray(model_outputs).argmax(axis=-1)
        else:
            predictions = model_outputs

        results: dict[str, float] = dict()
        for cfg in self.dataset_config.task.metrics:
            metric = self._metrics[cfg.name]

            # Some metrics can be computed on hardware accelerators. In this case we
            # start by setting the device to the same device as the model
            if cfg.compute_kwargs.get("device", None) == "auto":
                cfg.compute_kwargs["device"] = self.benchmark_config.device.type

            while True:
                try:
                    with HiddenPrints():
                        score_dict: dict[str, float] | None = metric.compute(
                            predictions=predictions,
                            references=labels,
                            **cfg.compute_kwargs,
                        )

                    # Clear the cache of the BERTScorer to avoid memory leaks
                    for attribute in METRIC_ATTRIBUTES_TAKING_UP_MEMORY:
                        if hasattr(metric, attribute):
                            delattr(metric, attribute)

                    clear_memory()
                    break
                except Exception as e:
                    # Clear the cache of the BERTScorer to avoid memory leaks
                    if hasattr(metric, "cached_bertscorer"):
                        del metric.cached_bertscorer
                        clear_memory()

                    oom_error = [
                        "CUDA out of memory",
                        "CUDA error",
                        "MPS backend out of memory",
                    ]
                    if not any(error in str(e) for error in oom_error):
                        raise InvalidBenchmark(str(e))

                    if cfg.compute_kwargs.get("batch_size", 1) > 1:
                        batch_size = cfg.compute_kwargs["batch_size"]
                        cfg.compute_kwargs["batch_size"] = batch_size // 2
                        logger.debug(
                            "Out of memory error occurred during the computation of "
                            f"the metric {cfg.pretty_name}. Reducing the batch size to "
                            f"{cfg.compute_kwargs['batch_size']}."
                        )
                    elif cfg.compute_kwargs.get("device", "cpu") != "cpu":
                        cfg.compute_kwargs["batch_size"] = 32
                        cfg.compute_kwargs["device"] = "cpu"
                        logger.debug(
                            "Out of memory error occurred during the computation of "
                            f"the metric {cfg.pretty_name}. Moving the computation to "
                            "the CPU."
                        )
                    else:
                        raise InvalidBenchmark(str(e))

            # The metric returns None if we are running on multi-GPU and the current
            # process is not the main process
            if score_dict is not None:
                scores = score_dict[cfg.results_key]
                if isinstance(scores, list):
                    scores = sum(scores) / len(scores)
                results[cfg.name] = scores

        return results

    def _extract_few_shot_examples(
        self, train_dataset: "Dataset", random_seed: int
    ) -> list[dict[str, Any]]:
        """Extract few-shot examples from the training dataset.

        Args:
            train_dataset:
                The training dataset.
            random_seed:
                The random seed to use when extracting the few-shot examples.

        Returns:
            The few-shot examples.
        """
        shuffled_train = train_dataset.shuffle(seed=random_seed)
        num_few_shots = self.dataset_config.num_few_shot_examples
        few_shot_examples: list[dict[str, Any]] = list()

        # We pick the few-shot examples one at a time rather than all at once since
        # we're working with a bootstrapped training dataset, meaning that it will have
        # duplicates. This ensures that we don't have any duplicates in the few-shot
        # examples
        while len(few_shot_examples) < num_few_shots:
            example = shuffled_train.select(range(1))[0]
            few_shot_examples.append(example)
            shuffled_train = shuffled_train.filter(
                lambda x: x["text"] != example["text"]
            )

        random.seed(random_seed)
        random.shuffle(few_shot_examples)
        return few_shot_examples

    def _apply_few_shot_prompt(
        self, examples: dict, few_shot_examples: list[dict], tokenizer: "Tokenizer"
    ) -> dict:
        """Apply a few-shot prompt to the examples.

        Args:
            examples:
                The examples to apply the prompt to.
            few_shot_examples:
                The examples to be included in the few-shot prompt.
            tokenizer:
                The tokenizer to use to encode the few-shot prompt.

        Returns:
            The examples with the few-shot prompt applied.
        """
        # Build the few-shot part of the prompt
        few_shot_prompts = [
            self.dataset_config.prompt_template.format(
                text=example["text"].replace("\n", " ").strip(),
                target_text=example["target_text"].replace("\n", " ").strip(),
            )
            for example in few_shot_examples
        ]
        prompt_prefix = ""
        if self.dataset_config.prompt_prefix:
            prompt_prefix = self.dataset_config.prompt_prefix + "\n\n"
        few_shot_prompt = prompt_prefix + "\n\n".join(few_shot_prompts)

        # Add the texts from the examples to the prompts. We remove newlines from the
        # examples as they have the special function to separate the few-shot examples
        # from one another
        new_prompts = [
            self.dataset_config.prompt_template.format(
                text=text.replace("\n", " ").strip(), target_text=""
            ).strip()
            for text in examples["text"]
        ]

        final_prompts = [
            few_shot_prompt + "\n\n" + new_prompt for new_prompt in new_prompts
        ]

        examples["text"] = final_prompts

        return examples

    def _extract_labels_from_generation(
        self,
        input_batch: dict[str, list],
        model_output: "ModelOutput",
        tokenizer: "Tokenizer",
    ) -> list[Any]:
        """Extract the predicted labels from the generated output.

        Args:
            input_batch:
                The input batch, where the keys are the feature names and the values
                are lists with the feature values.
            model_output:
                The raw generated output of the model.
            tokenizer:
                The tokenizer used together with the model.

        Returns:
            The predicted labels.
        """
        raw_predictions = extract_raw_predictions(
            generated_sequences=model_output.sequences, tokenizer=tokenizer
        )
        return raw_predictions
