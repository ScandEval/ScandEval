"""Abstract benchmarking dataset class."""

import logging
import sys
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Type

import evaluate
import torch
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import HfHubHTTPError, HFValidationError
from requests import RequestException
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel

from .exceptions import InvalidBenchmark
from .finetuning import finetune
from .generation import generate
from .model_config import get_model_config
from .model_loading import load_model
from .openai_models import OpenAIModel
from .scores import log_scores
from .speed_benchmark import benchmark_speed
from .tasks import SPEED
from .utils import (
    GENERATIVE_MODEL_TASKS,
    enforce_reproducibility,
    model_is_generative,
    should_prompts_be_stripped,
    unscramble,
)

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from numpy.random import Generator
    from transformers import PretrainedConfig
    from transformers.modeling_utils import ModelOutput

    from .config import BenchmarkConfig, DatasetConfig, ModelConfig
    from .protocols import GenerativeModel, Tokenizer
    from .types import Labels, Predictions, ScoreDict


logger = logging.getLogger(__package__)


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

    def benchmark(
        self,
        model_id: str,
        model: "GenerativeModel | None" = None,
        tokenizer: "Tokenizer | None" = None,
    ) -> "tuple[ScoreDict, dict[str, bool | int], GenerativeModel | None, Tokenizer | None]":
        """Benchmark a model.

        Args:
            model_id:
                The full Hugging Face Hub path to the pretrained transformer model. The
                specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id,
                and defaults to the latest version if not specified.
            model:
                The model to benchmark. If not provided, the model will be loaded.
            tokenizer:
                The tokenizer to use with the model. If not provided, the tokenizer will
                be loaded.

        Returns:
            A pair (scores, metadata_dict, model, tokenizer), with `scores` being a
            dictionary containing the scores, and `metadata_dict` being a dictionary
            containing various model metadata, such as the number of model parameters,
            the model's maximum sequence length and the size of the model's vocabulary.
            The keys in `score_dict` are 'raw' and 'total', with all the raw scores in
            the first dictionary and the aggregated scores in the second. The tokenizer
            and model are only not `None` if the model is generative.

        Raises:
            RuntimeError:
                If the extracted framework is not recognized.
        """
        model_config = get_model_config(
            model_id=model_id, benchmark_config=self.benchmark_config
        )

        # Set random seeds to enforce reproducibility of the randomly initialised
        # weights
        rng = enforce_reproducibility(framework=model_config.framework)

        if model is None or tokenizer is None:
            logger.info("Loading model and tokenizer...")
            model, tokenizer = load_model(
                model_config=model_config,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
            )

        benchmarking_generative_model = model_is_generative(model=model)

        # This happens when a local model is used, as we cannot fetch the model
        # metadata. Note that this is only the case if the model type is not any of the
        # ones hardcoded in `local.py`
        if model_config.task == "unknown":
            if benchmarking_generative_model:
                model_config.task = GENERATIVE_MODEL_TASKS[0]
            else:
                model_config.task = "fill-mask"

        metadata_dict = self._get_metadata(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            benchmarking_generative_model=benchmarking_generative_model,
        )

        if self.dataset_config.task != SPEED:
            train, val, tests = self._load_data(rng=rng)
            prepared_train, prepared_val, prepared_tests = self._load_prepared_data(
                train=train,
                val=val,
                tests=tests,
                model_config=model_config,
                hf_model_config=model.config,
                tokenizer=tokenizer,
                benchmarking_generative_model=benchmarking_generative_model,
            )

        # Set up progress bar
        itr = tqdm(
            iterable=range(self.benchmark_config.num_iterations),
            desc="Benchmarking",
            disable=not self.benchmark_config.progress_bar,
        )

        data_collator = self._load_data_collator(tokenizer=tokenizer, model=model)

        if self.dataset_config.task == SPEED:
            scores = benchmark_speed(
                itr=itr,
                tokenizer=tokenizer,
                model=model,
                model_config=model_config,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
            )
        elif benchmarking_generative_model:
            scores = generate(
                itr=itr,
                prepared_train=prepared_train,
                prepared_tests=prepared_tests,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics,
                extract_labels_fn=self._extract_labels_from_generation,
                benchmark_config=self.benchmark_config,
                dataset_config=self.dataset_config,
            )
        else:
            scores = finetune(
                itr=itr,
                train=train,
                val=val,
                tests=tests,
                prepared_train=prepared_train,
                prepared_val=prepared_val,
                prepared_tests=prepared_tests,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
                compute_metrics=self._compute_metrics,
                data_collator=data_collator,
                trainer_class=self._get_trainer_class(),
                evaluate_inputs_fn=self._get_evaluate_inputs,
                preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
            )

        all_scores = log_scores(
            dataset_name=self.dataset_config.pretty_name,
            metric_configs=self.dataset_config.task.metrics,
            scores=scores,
            model_id=model_config.model_id,
        )

        if benchmarking_generative_model:
            return all_scores, metadata_dict, model, tokenizer
        else:
            return all_scores, metadata_dict, None, None

    def _get_metadata(
        self,
        model_id: str,
        model: "PreTrainedModel | GenerativeModel",
        tokenizer: "Tokenizer",
        benchmarking_generative_model: bool,
    ) -> dict[str, int]:
        """Get metadata about the model.

        Args:
            model_id:
                The Hugging Face model ID.
            model:
                The model to get metadata about.
            tokenizer:
                The tokenizer to get metadata about.
            benchmarking_generative_model:
                Whether the model is a generative model.

        Returns:
            A dictionary containing metadata about the model, with the keys being the
            metadata names and the values being the metadata values.
        """
        api = HfApi()
        try:
            repo_info = api.repo_info(repo_id=model_id, repo_type="model")
            assert isinstance(repo_info, ModelInfo)
        except (RequestException, HFValidationError):
            repo_info = None

        if (
            repo_info is not None
            and hasattr(repo_info, "safetensors")
            and repo_info.safetensors is not None
            and "total" in repo_info.safetensors
        ):
            num_params = repo_info.safetensors["total"]
        elif hasattr(model.config, "num_params"):
            num_params = model.config.num_params
        elif isinstance(model, PreTrainedModel):
            num_params = sum(p.numel() for p in model.parameters())
        else:
            num_params = -1

        if hasattr(model.config, "model_max_length"):
            max_seq_length = getattr(model.config, "model_max_length")
        elif hasattr(model.config, "max_sequence_length"):
            max_seq_length = getattr(model.config, "max_sequence_length")
        elif hasattr(
            tokenizer, "model_max_length"
        ) and tokenizer.model_max_length < int(1e30):
            max_seq_length = getattr(tokenizer, "model_max_length")
        else:
            max_seq_length = -1

        # If the model is a generative model then we have subtracted the generation
        # length from the maximum length to allow it to keep generating. But for the
        # model metadata we want to know the maximum length, so we add the generation
        # length back on here
        if max_seq_length >= 0 and benchmarking_generative_model:
            max_seq_length += self.dataset_config.max_generated_tokens

            # If the model is an OpenAI chat model then we add on 7 extra tokens, as
            # these are part of the chat prompt and was removed from the sequence
            # length
            if isinstance(model, OpenAIModel) and model.is_chat_model:
                max_seq_length += 7

        if hasattr(model.config, "vocab_size"):
            vocab_size = getattr(model.config, "vocab_size")
        elif hasattr(tokenizer, "vocab_size"):
            vocab_size = getattr(tokenizer, "vocab_size")
        else:
            vocab_size = -1

        # Store the metadata in a dictionary
        metadata_dict = dict(
            num_model_parameters=num_params,
            max_sequence_length=max_seq_length,
            vocabulary_size=vocab_size,
            generative=benchmarking_generative_model,
            # TODO: This will be changed when we support finetuning of generative models
            few_shot=True,
            validation_split=self.benchmark_config.only_validation_split,
        )

        # Log the metadata
        logging_msg: str = ""
        if num_params < 0:
            logging_msg += "The model has an unknown number of parameters, "
        else:
            logging_msg += f"The model has {num_params:,} parameters, "
        if vocab_size < 0:
            logging_msg += "an unknown vocabulary size, "
        else:
            logging_msg += f"a vocabulary size of {vocab_size:,}, "
        if max_seq_length < 0:
            logging_msg += "and an unknown maximum sequence length."
        else:
            logging_msg += f"and a maximum sequence length of {max_seq_length:,}."
        logger.info(logging_msg)

        return metadata_dict

    def _load_data(
        self, rng: "Generator"
    ) -> tuple["Dataset", "Dataset", list["Dataset"]]:
        """Load the raw bootstrapped datasets.

        Args:
            rng:
                The random number generator to use.

        Returns:
            A tuple containing the training, validation and test datasets.
        """
        # Download dataset from the HF Hub
        try:
            dataset_dict = load_dataset(
                path=self.dataset_config.huggingface_id,
                cache_dir=self.benchmark_config.cache_dir,
                token=unscramble("HjccJFhIozVymqXDVqTUTXKvYhZMTbfIjMxG_"),
            )
        except HfHubHTTPError:
            raise InvalidBenchmark("The Hugging Face Hub seems to be down.")

        # If the dataset turns out not to be a DatasetDict, then we raise an error
        if not isinstance(dataset_dict, DatasetDict):
            raise InvalidBenchmark(
                f"Expected `dataset_dict` to be a `DatasetDict`, but got "
                f"{type(dataset_dict)}."
            )

        # Remove all other keys than 'train', 'val' and 'test'
        dataset_dict = DatasetDict(
            {key: dataset_dict[key] for key in ["train", "val", "test"]}
        )

        # Process the datasets
        dataset_dict = self._process_data(dataset_dict)

        # Extract the dataset splits
        train = dataset_dict["train"]
        val = dataset_dict["val"]
        test = dataset_dict["test"]

        if self.benchmark_config.only_validation_split:
            test = val

        # Remove empty examples from the datasets
        for text_feature in ["tokens", "text"]:
            if text_feature in train.features:
                train = train.filter(lambda x: len(x[text_feature]) > 0)
                val = val.filter(lambda x: len(x[text_feature]) > 0)
                test = test.filter(lambda x: len(x[text_feature]) > 0)

        # If we are testing then truncate the test set
        if hasattr(sys, "_called_from_test"):
            test = test.select(range(1))

        # Bootstrap the test set
        test_bidxs = rng.integers(
            0, len(test), size=(self.benchmark_config.num_iterations, len(test))
        )
        tests = [test.select(test_bidxs[idx]) for idx in range(test_bidxs.shape[0])]

        return train, val, tests

    def _load_prepared_data(
        self,
        train: "Dataset",
        val: "Dataset",
        tests: list["Dataset"],
        model_config: "ModelConfig",
        hf_model_config: "PretrainedConfig",
        tokenizer: "Tokenizer",
        benchmarking_generative_model: bool,
    ) -> tuple["Dataset", "Dataset", list["Dataset"]]:
        """Load the data and prepare it for training.

        Args:
            train:
                The raw training dataset.
            val:
                The raw validation dataset.
            tests:
                The raw bootstrapped test datasets.
            model_config:
                The model configuration.
            hf_model_config:
                The Hugging Face model configuration.
            tokenizer:
                The tokenizer.
            benchmarking_generative_model:
                Whether the model is a generative model.

        Returns:
            A tuple containing the prepared training, validation and test datasets.
        """
        # Set up the preprocessing parameters
        preprocess_params: dict[str, Any] = dict(
            hf_model_config=hf_model_config,
            model_config=model_config,
            tokenizer=tokenizer,
            generative_model=benchmarking_generative_model,
        )

        # Prepare the train and validation datasets
        with tqdm(
            total=2 + self.benchmark_config.num_iterations,
            desc="Preprocessing data splits",
            disable=hasattr(sys, "_called_from_test"),
        ) as pbar:
            # When evaluating generative models we only need the test split, so
            # there's no need to prepare the train split
            try:
                prepared_train = train
                if not benchmarking_generative_model:
                    prepared_train = self._preprocess_data(
                        train, split="train", **preprocess_params
                    )
                pbar.update(1)
            except ValueError:
                raise InvalidBenchmark(
                    "Preprocessing of the training dataset could not be done."
                )

            # When evaluating generative models we only need the test split, so
            # there's no need to prepare the validation split
            try:
                prepared_val = val
                if not benchmarking_generative_model:
                    prepared_val = self._preprocess_data(
                        val, split="val", **preprocess_params
                    )
                pbar.update(1)
            except ValueError:
                raise InvalidBenchmark(
                    "Preprocessing of the validation dataset could not be done."
                )

            try:
                prepared_tests: list["Dataset"] = list()
                for itr_idx, test in enumerate(tests):
                    if benchmarking_generative_model:
                        itr_seed = 4242 + itr_idx
                        few_shot_examples = self._extract_few_shot_examples(
                            train_dataset=train, random_seed=itr_seed
                        )
                        few_shot_fn = partial(
                            self._apply_few_shot_prompt,
                            few_shot_examples=few_shot_examples,
                            tokenizer=tokenizer,
                        )
                        test = test.map(
                            few_shot_fn,
                            batched=True,
                            load_from_cache_file=False,
                            keep_in_memory=True,
                        )

                        # NOTE: This applies the model's chat template if one is
                        # available. However, all experiments have shown this to reduce
                        # overall performance, so it is left out for now.
                        # test = test.map(
                        #     function=lambda x: dict(
                        #         text=convert_prompt_to_instruction(
                        #             prompt=x["text"], tokenizer=tokenizer
                        #         )
                        #     ),
                        #     load_from_cache_file=False,
                        #     keep_in_memory=True,
                        # )

                        # Determine if we should strip the prompts. This is the case if
                        # the tokenizer needs to include the space as part of the label
                        # token
                        labels_to_be_generated = list(
                            self.dataset_config.prompt_label_mapping.values()
                        )
                        strip_prompts = should_prompts_be_stripped(
                            labels_to_be_generated=labels_to_be_generated,
                            tokenizer=tokenizer,
                        )
                        if strip_prompts:
                            test = test.map(
                                lambda x: dict(text=x["text"].strip()),
                                load_from_cache_file=False,
                                keep_in_memory=True,
                            )

                    prepared_test = self._preprocess_data(
                        test, split="test", **preprocess_params
                    )

                    if benchmarking_generative_model:
                        prepared_test = prepared_test.map(
                            lambda examples: dict(
                                text=tokenizer.batch_decode(
                                    sequences=examples["input_ids"],
                                    skip_special_tokens=True,
                                )
                            ),
                            batched=True,
                            load_from_cache_file=False,
                            keep_in_memory=True,
                        )

                    prepared_tests.append(prepared_test)
                    pbar.update(1)
            except ValueError:
                raise InvalidBenchmark(
                    "Preprocessing of the test dataset could not be done."
                )

        return prepared_train, prepared_val, prepared_tests

    def _preprocess_logits_for_metrics(
        self, model_outputs: torch.Tensor | tuple, labels: torch.Tensor
    ) -> torch.Tensor | tuple:
        """Ensure that only the logits are returned from the model.

        This is to avoid memory issues when the model returns hidden states as well.

        Args:
            model_outputs:
                The raw model outputs.
            labels:
                The ground truth labels.

        Returns:
            The preprocessed logits.
        """
        if isinstance(model_outputs, tuple) and isinstance(
            model_outputs[0], torch.Tensor
        ):
            model_output_tensors = [
                model_output
                for model_output in model_outputs
                if isinstance(model_output, torch.Tensor)
            ]
            if len(model_output_tensors) == 1:
                return model_output_tensors[0]
            return tuple(model_output_tensors)
        else:
            return model_outputs

    def __call__(self, *args, **kwargs):
        """Call the benchmark method. See `benchmark` for details."""
        return self.benchmark(*args, **kwargs)

    def _process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Process the data.

        Args:
            dataset_dict:
                The dataset dictionary.

        Returns:
            The processed dataset dictionary.
        """
        return dataset_dict

    def _get_trainer_class(self) -> Type[Trainer]:
        """Returns the trainer class to use."""
        return Trainer

    def _get_evaluate_inputs(
        self, dataset: "Dataset", prepared_dataset: "Dataset", metric_key_prefix: str
    ) -> dict[str, Any]:
        """Returns the inputs to the `Trainer.evaluate` method.

        Args:
            dataset:
                The raw dataset.
            prepared_dataset:
                The prepared dataset.
            metric_key_prefix:
                The prefix to use for the metric keys.
        """
        return dict(eval_dataset=prepared_dataset, metric_key_prefix=metric_key_prefix)

    @abstractmethod
    def _preprocess_data(self, dataset: "Dataset", **kwargs) -> "Dataset":
        """Preprocess a dataset.

        Args:
            dataset:
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            The preprocessed dataset.
        """
        pass

    @abstractmethod
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
                A pretrained model. Can be None if the model is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            The data collator.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
