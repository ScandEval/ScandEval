"""Abstract benchmarking dataset class."""

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import evaluate
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from huggingface_hub.utils._errors import HfHubHTTPError
from tqdm.auto import tqdm
from transformers import PretrainedConfig
from transformers.modeling_utils import ModelOutput, PreTrainedModel

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .dataset_tasks import SPEED
from .exceptions import InvalidBenchmark
from .finetuning import finetune
from .generation import generate
from .model_config import get_model_config
from .model_loading import load_model
from .model_setups import GenerativeModel, Tokenizer
from .openai_models import OpenAIModel
from .scores import log_scores
from .speed_benchmark import benchmark_speed
from .types import SCORE_DICT
from .utils import enforce_reproducibility, model_is_generative

logger = logging.getLogger(__package__)


class BenchmarkDataset(ABC):
    """Abstract benchmarking dataset class.

    Args:
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Attributes:
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.
    """

    def __init__(
        self, dataset_config: DatasetConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Initialise the dataset.

        Args:
            dataset_config (DatasetConfig):
                The configuration for the dataset.
            benchmark_config (BenchmarkConfig):
                The configuration for the benchmark.
        """
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self._metrics = {
            metric_cfg.name: evaluate.load(
                path=metric_cfg.huggingface_id,
                cache_dir=self.benchmark_config.cache_dir,
            )
            if metric_cfg.huggingface_id != ""
            else None
            for metric_cfg in dataset_config.task.metrics
        }

        # Set logging level based on verbosity
        logging_level = logging.DEBUG if self.benchmark_config.verbose else logging.INFO
        logger.setLevel(logging_level)

    def benchmark(self, model_id: str) -> tuple[SCORE_DICT, dict[str, int]]:
        """Benchmark a model.

        Args:
            model_id (str):
                The full Hugging Face Hub path to the pretrained transformer model. The
                specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id,
                and defaults to the latest version if not specified.

        Returns:
            pair of dicts:
                A pair (scores, metadata_dict), with `scores` being a dictionary
                containing the scores, and `metadata_dict` being a dictionary
                containing various model metadata, such as the number of model
                parameters, the model's maximum sequence length and the size of the
                model's vocabulary. The keys in `score_dict` are 'raw' and 'total',
                with all the raw scores in the first dictionary and the aggregated
                scores in the second.

        Raises:
            RuntimeError:
                If the extracted framework is not recognized.
        """
        model_config = get_model_config(
            model_id=model_id, benchmark_config=self.benchmark_config
        )

        rng = enforce_reproducibility(framework=model_config.framework)

        tokenizer, model = load_model(
            model_config=model_config,
            dataset_config=self.dataset_config,
            benchmark_config=self.benchmark_config,
        )

        # This happens when a local model is used, as we cannot fetch the model metadata
        if model_config.task == "unknown":
            if model_is_generative(model=model):
                model_config.task = "text-generation"
            else:
                model_config.task = "fill-mask"

        metadata_dict = self._get_metadata(model=model, tokenizer=tokenizer)

        # Set variable with number of iterations
        num_iter = 10 if not self.benchmark_config.testing else 5

        if self.dataset_config.task.name != SPEED:
            train, val, tests = self._load_data(num_iter=num_iter, rng=rng)
            prepared_train, prepared_val, prepared_tests = self._load_prepared_data(
                train=train,
                val=val,
                tests=tests,
                model_config=model_config,
                hf_model_config=model.config,
                tokenizer=tokenizer,
                generative_model=model_is_generative(model=model),
            )

        # Set up progress bar
        itr = tqdm(
            iterable=range(num_iter),
            desc="Benchmarking",
            disable=not self.benchmark_config.progress_bar,
        )

        data_collator = self._load_data_collator(tokenizer=tokenizer, model=model)

        if self.dataset_config.task.name == SPEED:
            scores = benchmark_speed(
                itr=itr,
                tokenizer=tokenizer,
                model=model,
                model_config=model_config,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
            )
        elif model_is_generative(model=model):
            scores = generate(
                itr=itr,
                train=train,
                val=val,
                tests=tests,
                prepared_train=prepared_train,
                prepared_val=prepared_val,
                prepared_tests=prepared_tests,
                model=model,
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
            )

        all_scores = log_scores(
            dataset_name=self.dataset_config.pretty_name,
            metric_configs=self.dataset_config.task.metrics,
            scores=scores,
            model_id=model_config.model_id,
        )

        return all_scores, metadata_dict

    def _get_metadata(
        self,
        model: PreTrainedModel | GenerativeModel,
        tokenizer: Tokenizer,
    ) -> dict[str, int]:
        """Get metadata about the model.

        Args:
            model (PreTrainedModel or GenerativeModel):
                The model to get metadata about.
            tokenizer (Tokenizer):
                The tokenizer to get metadata about.

        Returns:
            dict[str, int]:
                A dictionary containing metadata about the model, with the keys being
                the metadata names and the values being the metadata values.
        """
        if hasattr(model.config, "num_params"):
            num_params = model.config.num_params
        elif isinstance(model, PreTrainedModel):
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            num_params = -1

        if hasattr(model.config, "model_max_length"):
            max_seq_length = getattr(model.config, "model_max_length")
        elif hasattr(tokenizer, "model_max_length"):
            max_seq_length = getattr(tokenizer, "model_max_length")
        else:
            max_seq_length = -1

        # If the model is a generative model then we have subtracted the generation
        # length from the maximum length to allow it to keep generating. But for the
        # model metadata we want to know the maximum length, so we add the generation
        # length back on here
        if model_is_generative(model=model):
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
        self,
        num_iter: int,
        rng: np.random.Generator,
    ) -> tuple[Dataset, Dataset, list[Dataset]]:
        """Load the raw bootstrapped datasets.

        Args:
            num_iter (int):
                The number of iterations to run.
            rng (np.random.Generator):
                The random number generator to use.

        Returns:
            tuple[Dataset, Dataset, list[Dataset]]:
                A tuple containing the training, validation and test datasets.
        """
        # Download dataset from the HF Hub
        try:
            dataset_dict = load_dataset(
                path=self.dataset_config.huggingface_id,
                cache_dir=self.benchmark_config.cache_dir,
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

        # TEMP
        test = val

        # Remove empty examples from the datasets
        for text_feature in ["tokens", "text"]:
            if text_feature in train.features:
                train = train.filter(lambda x: len(x[text_feature]) > 0)
                val = val.filter(lambda x: len(x[text_feature]) > 0)
                test = test.filter(lambda x: len(x[text_feature]) > 0)

        # If we are testing then truncate the test set
        if self.benchmark_config.testing:
            test = test.select(range(128))

        # Bootstrap the test set
        test_bidxs = rng.integers(0, len(test), size=(num_iter, len(test)))
        tests = [test.select(test_bidxs[idx]) for idx in range(test_bidxs.shape[0])]

        return train, val, tests

    def _load_prepared_data(
        self,
        train: Dataset,
        val: Dataset,
        tests: list[Dataset],
        model_config: ModelConfig,
        hf_model_config: PretrainedConfig,
        tokenizer: Tokenizer,
        generative_model: bool,
    ) -> tuple[Dataset, Dataset, list[Dataset]]:
        """Load the data and prepare it for training.

        Args:
            train (Dataset):
                The raw training dataset.
            val (Dataset):
                The raw validation dataset.
            tests (list[Dataset]):
                The raw bootstrapped test datasets.
            model_config (ModelConfig):
                The model configuration.
            hf_model_config (PretrainedConfig):
                The Hugging Face model configuration.
            tokenizer (Tokenizer):
                The tokenizer.
            generative_model (bool):
                Whether the model is a generative model.

        Returns:
            tuple[Dataset, Dataset, list[Dataset]]:
                A tuple containing the prepared training, validation and test datasets.
        """
        # Set up the preprocessing parameters
        preprocess_params: dict[str, Any] = dict(
            hf_model_config=hf_model_config,
            model_config=model_config,
            tokenizer=tokenizer,
            generative_model=generative_model,
        )

        # Prepare the train and validation datasets
        try:
            with tqdm(total=12, desc="Preprocessing data splits", leave=False) as pbar:
                prepared_train = train
                if not generative_model:
                    prepared_train = self._preprocess_data(
                        train, split="train", **preprocess_params
                    )
                pbar.update(1)

                prepared_val = val
                if not generative_model:
                    prepared_val = self._preprocess_data(
                        val, split="val", **preprocess_params
                    )
                pbar.update(1)

                prepared_tests: list[Dataset] = list()
                for itr_idx, test in enumerate(tests):
                    if generative_model:
                        itr_seed = 4242 + itr_idx
                        few_shot_examples = self._extract_few_shot_examples(
                            train_dataset=train, random_seed=itr_seed
                        )
                        few_shot_fn = partial(
                            self._apply_few_shot_prompt,
                            few_shot_examples=few_shot_examples,
                        )
                        test = test.map(
                            few_shot_fn, batched=True, load_from_cache_file=False
                        )
                    prepared_test = self._preprocess_data(
                        test, split="test", **preprocess_params
                    )
                    prepared_tests.append(prepared_test)

                    pbar.update(1)
        except ValueError:
            raise InvalidBenchmark(
                "Preprocessing of the training and validation datasets could not be "
                "done."
            )

        return prepared_train, prepared_val, prepared_tests

    def __call__(self, *args, **kwargs):
        return self.benchmark(*args, **kwargs)

    def _process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Process the data.

        Args:
            dataset_dict (DatasetDict):
                The dataset dictionary.

        Returns:
            DatasetDict:
                The processed dataset dictionary.
        """
        return dataset_dict

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
        """Preprocess a dataset.

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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
