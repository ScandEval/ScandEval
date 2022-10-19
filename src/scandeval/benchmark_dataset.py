"""Abstract benchmarking dataset class."""

import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset, load_metric
from numpy._typing import NDArray
from tqdm.auto import tqdm
from transformers import logging as tf_logging
from transformers.trainer import Trainer
from transformers.trainer_callback import (
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames, TrainingArguments
from transformers.utils.import_utils import is_torch_tpu_available

from .callbacks import NeverLeaveProgressCallback
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .hf_hub import get_model_config
from .model_loading import load_model
from .protocols import DataCollator, Model, Tokenizer
from .scores import log_scores
from .training_args_with_mps_support import TrainingArgumentsWithMPSSupport
from .utils import clear_memory, enforce_reproducibility, handle_error

# Set up logger
logger = logging.getLogger(__name__)


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
            metric_cfg.name: load_metric(metric_cfg.huggingface_id)
            for metric_cfg in dataset_config.task.metrics
        }

    # TODO: Cache this
    def benchmark(
        self,
        model_id: str,
    ) -> Dict[str, Union[Dict[str, float], Dict[str, List[Dict[str, float]]]]]:
        """Benchmark a model.

        Args:
            model_id (str):
                The full Hugging Face Hub path to the pretrained transformer model. The
                specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id.

        Returns:
            dict:
                The keys in the dict are 'raw' and 'total', with all the raw scores in
                the first dictionary and the aggregated scores in the second.

        Raises:
            RuntimeError:
                If the extracted framework is not recognized.
        """
        # Fetch the model config
        model_config = get_model_config(
            model_id=model_id, benchmark_config=self.benchmark_config
        )

        # Set random seeds to enforce reproducibility of the randomly initialised
        # weights
        rng = enforce_reproducibility(framework=model_config.framework)

        # Load the model
        tokenizer, model = load_model(
            model_id=model_id,
            revision=model_config.revision,
            supertask=self.dataset_config.task.supertask,
            num_labels=self.dataset_config.num_labels,
            id2label=self.dataset_config.id2label,
            label2id=self.dataset_config.label2id,
            from_flax=model_config.framework == "jax",
            use_auth_token=self.benchmark_config.use_auth_token,
            cache_dir=self.benchmark_config.cache_dir,
        )

        # Log the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of model parameters: {num_params:,}")

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        # Load the data
        train, val, test = self._load_data()

        # Preprocess the datasets
        try:
            params = dict(framework="pytorch", config=model.config, tokenizer=tokenizer)
            train = self._preprocess_data(train, **params)
            val = self._preprocess_data(val, **params)
            test = self._preprocess_data(test, **params)
        except ValueError:
            raise InvalidBenchmark("Preprocessing of the dataset could not be done.")

        # Set variable with number of iterations
        num_iter = 10 if not self.benchmark_config.testing else 2

        # Get bootstrap sample indices
        test_bidxs = rng.integers(0, len(test), size=(num_iter, len(test)))

        # Get bootstrapped datasets
        tests = [
            Dataset.from_dict(test[test_bidxs[idx]])
            for idx in range(test_bidxs.shape[0])
        ]

        # Get the training arguments
        training_args = self._get_training_args()

        # Set up progress bar
        itr = tqdm(
            iterable=range(num_iter),
            desc="Benchmarking",
            disable=not self.benchmark_config.progress_bar,
        )

        scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for idx in itr:
            while True:
                itr_scores = self._benchmark_single_iteration(
                    idx=idx,
                    model_config=model_config,
                    train=train,
                    val=val,
                    tests=tests,
                    data_collator=data_collator,
                    training_args=training_args,
                    tokenizer=tokenizer if idx == 0 else None,
                    model=model if idx == 0 else None,
                )

                # If the iteration was successful then break the loop
                if isinstance(itr_scores, dict):
                    break

                # Otherwise we encountered an error, so we have to deal with it and try
                # again
                else:
                    bs = training_args.per_device_train_batch_size
                    ga = training_args.gradient_accumulation_steps
                    bs, ga = handle_error(
                        e=itr_scores,
                        per_device_train_batch_size=bs,
                        gradient_accumulation_steps=ga,
                    )

                    training_args.per_device_train_batch_size = bs
                    training_args.per_device_eval_batch_size = bs
                    training_args.gradient_accumulation_steps = ga

            if "train" in itr_scores:
                scores["train"].append(itr_scores["train"])
            scores["test"].append(itr_scores["test"])

        all_scores = log_scores(
            dataset_name=self.dataset_config.pretty_name,
            metric_configs=self.dataset_config.task.metrics,
            scores=scores,
            model_id=model_config.model_id,
        )

        # Garbage collection, to avoid memory issues
        del model
        del tokenizer
        clear_memory()

        return all_scores

    def _get_training_args(self) -> TrainingArguments:

        # Set the logging strategy
        if self.benchmark_config.verbose:
            logging_strategy = IntervalStrategy.STEPS
        else:
            logging_strategy = IntervalStrategy.NO

        # Use 16-bit floating point numbers if CUDA is available and TPU is not
        fp16 = torch.cuda.is_available() and not is_torch_tpu_available()

        # Initialise training arguments
        training_args = TrainingArgumentsWithMPSSupport(
            output_dir=self.benchmark_config.cache_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
            logging_strategy=logging_strategy,
            save_strategy=IntervalStrategy.STEPS,
            eval_steps=30,
            logging_steps=30,
            save_steps=30,
            max_steps=10_000 if not self.benchmark_config.testing else 2,
            report_to=None,
            save_total_limit=1,
            per_device_train_batch_size=32 if not self.benchmark_config.testing else 1,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.01,
            gradient_accumulation_steps=1,
            load_best_model_at_end=True,
            optim=OptimizerNames.ADAMW_TORCH,
            seed=4242,
            no_cuda=self.benchmark_config.testing,
            fp16=fp16,
        )

        # Manually set `disable_tqdm` to `False` if `progress_bar` is `True`
        if self.benchmark_config.progress_bar:
            training_args.disable_tqdm = False

        return training_args

    def _load_data(self):

        # Download dataset from the HF Hub
        dataset_dict = load_dataset(
            path=self.dataset_config.huggingface_id,
            use_auth_token=self.benchmark_config.use_auth_token,
            cache_dir=self.benchmark_config.cache_dir,
        )

        # If the dataset turns out not to be a DatasetDict, then we raise an error
        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError(
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

        # Remove empty examples from the datasets
        if "tokens" in train.features:
            train = train.filter(lambda x: len(x["tokens"]) > 0)
            val = val.filter(lambda x: len(x["tokens"]) > 0)
            test = test.filter(lambda x: len(x["tokens"]) > 0)
        elif "doc" in train.features:
            train = train.filter(lambda x: len(x["doc"]) > 0)
            val = val.filter(lambda x: len(x["doc"]) > 0)
            test = test.filter(lambda x: len(x["doc"]) > 0)

        # If we are testing then truncate the test set
        if self.benchmark_config.testing:
            test = test.select(range(128))

        return train, val, test

    def _benchmark_single_iteration(
        self,
        idx: int,
        model_config: ModelConfig,
        train: Dataset,
        val: Dataset,
        tests: Sequence[Dataset],
        data_collator: DataCollator,
        training_args: TrainingArguments,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[Model] = None,
    ) -> Union[Dict[str, Dict[str, float]], Exception]:
        """Run a single iteration of a benchmark.

        Args:
            idx (int):
                The index of the current iteration.
            model_config (ModelConfig):
                The model configuration.
            train (Dataset):
                The training dataset.
            val (Dataset):
                The validation dataset.
            tests (list of Dataset):
                The test datasets.
            data_collator (DataCollator):
                The data collator.
            training_args (TrainingArguments):
                The training arguments.
            tokenizer (Tokenizer or None, optional):
                The tokenizer to use in the benchmark. If None then a new tokenizer
                will be loaded. Defaults to None.
            model (Model or None, optional):
                The model to use in the benchmark. If None then a new model will be
                loaded. Defaults to None.

        Returns:
            dict or Exception:
                A dictionary containing the scores for the current iteration, with keys
                `train` and `test`. If an exception is raised, then the exception is
                returned.
        """
        # Set transformers logging back to error
        tf_logging.set_verbosity(logging.CRITICAL)

        scores: Dict[str, Dict[str, float]] = dict()
        try:
            # Set random seeds to enforce reproducibility of the randomly
            # initialised weights
            training_args.seed = 4242 + idx
            random.seed(4242 + idx)
            np.random.seed(4242 + idx)
            torch.manual_seed(4242 + idx)
            torch.cuda.manual_seed_all(4242 + idx)

            # Reinitialise a new model
            if tokenizer is None or model is None:
                tokenizer, model = load_model(
                    model_id=model_config.model_id,
                    revision=model_config.revision,
                    supertask=self.dataset_config.task.supertask,
                    num_labels=self.dataset_config.num_labels,
                    label2id=self.dataset_config.label2id,
                    id2label=self.dataset_config.id2label,
                    from_flax=model_config.framework == "jax",
                    use_auth_token=self.benchmark_config.use_auth_token,
                    cache_dir=self.benchmark_config.cache_dir,
                )

            # Initialise compute_metrics function
            compute_metrics = partial(
                self._compute_metrics, id2label=model.config.id2label
            )

            # Initialise early stopping callback
            early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

            # Disable logging from trainer.py
            logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)

            # Initialise Trainer
            trainer_args = dict(
                model=model,
                args=training_args,
                train_dataset=train,
                eval_dataset=val,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping],
            )
            trainer = Trainer(**trainer_args)

            # Set transformers logging back to error
            tf_logging.set_verbosity(logging.CRITICAL)

            # Remove trainer logging if not in verbose mode
            if not self.benchmark_config.verbose:
                trainer.log = lambda logs: None

            # Remove the callback which prints the scores after each
            # evaluation
            if not self.benchmark_config.verbose:
                trainer.remove_callback(PrinterCallback)

            # Remove the progress bar callback
            trainer.remove_callback(ProgressCallback)

            # Add the custom progress callback if `progress_bar` is True
            if self.benchmark_config.progress_bar:
                trainer.add_callback(NeverLeaveProgressCallback)

            # Finetune the model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                trainer.train()

            # Log training scores and save the state
            if self.benchmark_config.evaluate_train:
                train_scores = trainer.evaluate(train, metric_key_prefix="train")
                scores["train"] = train_scores

            # Set up a progress bar for the test datasets if we are not
            # finetuning
            test_itr = [tests[idx]]

            # Log test scores
            for dataset in test_itr:
                test_scores = trainer.evaluate(dataset, metric_key_prefix="test")

                scores["test"] = test_scores

            # Return the scores
            return scores

        except (RuntimeError, ValueError, IndexError) as e:
            try:
                del model
            except UnboundLocalError:
                pass
            try:
                del tokenizer
            except UnboundLocalError:
                pass
            clear_memory()
            return e

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
        pass

    @abstractmethod
    def _load_data_collator(
        self, tokenizer: Optional[Tokenizer] = None
    ) -> DataCollator:
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        pass

    def _compute_metrics(
        self,
        probabilities_and_labels: Tuple[NDArray[np.float_], NDArray[np.int_]],
        id2label: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            probabilities_and_labels (pair of arrays):
                The first array contains the probability predictions and the second
                array contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        probabilities, labels = probabilities_and_labels
        predictions: NDArray[np.int_] = probabilities.argmax(axis=-1)
        results: Dict[str, float] = dict()
        for cfg in self.dataset_config.task.metrics:
            metric = self._metrics[cfg.name]
            score_dict: Union[None, Dict[str, float]] = metric.compute(
                predictions=predictions,
                references=labels,
                **cfg.compute_kwargs,
            )
            if score_dict is not None:
                scores = score_dict[cfg.results_key]
                results[cfg.name] = scores
        return results
