"""Abstract benchmarking dataset class."""

import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import evaluate
import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from tqdm.auto import tqdm
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import (
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames, TrainingArguments

from .callbacks import NeverLeaveProgressCallback
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .hf_hub import get_model_config
from .model_loading import load_model
from .scores import log_scores
from .speed_benchmark import benchmark_speed
from .types import SCORE_DICT
from .utils import (
    block_terminal_output,
    clear_memory,
    enforce_reproducibility,
    handle_error,
)

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
            metric_cfg.name: evaluate.load(
                metric_cfg.huggingface_id, cache_dir=self.benchmark_config.cache_dir
            )
            if metric_cfg.huggingface_id != ""
            else None
            for metric_cfg in dataset_config.task.metrics
        }

    def benchmark(  # noqa
        self,
        model_id: str,
    ) -> Tuple[SCORE_DICT, Dict[str, int]]:
        """Benchmark a model.

        Args:
            model_id (str):
                The full Hugging Face Hub path to the pretrained transformer model. The
                specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id,
                and defaults to the latest version if not specified.

        Returns:
            pair of dicts:
                A pair (score_dict, metadata_dict), with `score_dict` being a
                dictionary containing the scores, and `metadata_dict` being a
                dictionary containing various model metadata, such as the number of
                model parameters, the model's maximum sequence length and the size of
                the model's vocabulary. The keys in `score_dict` are 'raw' and 'total',
                with all the raw scores in the first dictionary and the aggregated
                scores in the second.

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
            model_id=model_config.model_id,
            revision=model_config.revision,
            supertask=self.dataset_config.task.supertask,
            num_labels=self.dataset_config.num_labels,
            id2label=self.dataset_config.id2label,
            label2id=self.dataset_config.label2id,
            from_flax=model_config.framework == "jax",
            use_auth_token=self.benchmark_config.use_auth_token,
            cache_dir=self.benchmark_config.cache_dir,
        )

        # Get the metadata
        metadata_dict = self._get_metadata(model=model, tokenizer=tokenizer)

        # Set variable with number of iterations
        num_iter = 10 if not self.benchmark_config.testing else 5

        # Set up progress bar
        itr = tqdm(
            iterable=range(num_iter),
            desc="Benchmarking",
            disable=not self.benchmark_config.progress_bar,
        )

        # If we are running the speed estimation benchmark then call that directly
        if self.dataset_config.task.name == "speed":
            all_scores = benchmark_speed(
                itr=itr,
                tokenizer=tokenizer,
                model=model,
                model_config=model_config,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
            )
            return all_scores, metadata_dict

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        # Load the data
        train, val, test = self._load_data()

        # Get bootstrap sample indices
        test_bidxs = rng.integers(0, len(test), size=(num_iter, len(test)))

        # Get bootstrapped datasets
        tests = [test.select(test_bidxs[idx]) for idx in range(test_bidxs.shape[0])]

        # Set up the preprocessing parameters
        preprocess_params = dict(
            framework="pytorch", config=model.config, tokenizer=tokenizer
        )

        # Prepare the train and validation datasets
        try:
            prepared_train = self._preprocess_data(
                train, split="train", **preprocess_params
            )
            prepared_val = self._preprocess_data(val, split="val", **preprocess_params)
        except ValueError:
            raise InvalidBenchmark(
                "Preprocessing of the training and validation datasets could not be "
                "done."
            )

        # Initialise the `scores` dictionary
        scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        bs: int = self.benchmark_config.batch_size
        ga: int = 32 // bs
        for idx in itr:

            # Set variable that tracks whether we need to initialize new models in the
            # `_benchmark_single_iteration` call
            model_already_initialized = idx == 0

            # Clear memory after first iteration
            if not model_already_initialized:
                try:
                    del model
                except UnboundLocalError:
                    pass
                try:
                    del tokenizer
                except UnboundLocalError:
                    pass
                clear_memory()

            while True:

                # Get the boostrapped dataset for this iteration
                test = tests[idx]

                # Prepare the test dataset
                try:
                    prepared_test = self._preprocess_data(
                        test, split="test", **preprocess_params
                    )
                except ValueError:
                    raise InvalidBenchmark(
                        "Preprocessing of the test dataset could not be done."
                    )

                # Re-block terminal output, as it gets unblocked by the `transformers`
                # package before training
                block_terminal_output()

                # Get the training arguments
                training_args = self._get_training_args(iteration_idx=idx)

                # Set the correct batch size and gradient accumulation
                training_args.per_device_train_batch_size = bs
                training_args.per_device_eval_batch_size = bs
                training_args.gradient_accumulation_steps = ga

                itr_scores = self._benchmark_single_iteration(
                    iteration_idx=idx,
                    model_config=model_config,
                    train=train,
                    prepared_train=prepared_train,
                    prepared_val=prepared_val,
                    test=test,
                    prepared_test=prepared_test,
                    data_collator=data_collator,
                    training_args=training_args,
                    tokenizer=tokenizer if model_already_initialized else None,
                    model=model if model_already_initialized else None,
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

                    # Clear memory, to avoid memory issues
                    try:
                        del model
                    except UnboundLocalError:
                        pass
                    try:
                        del tokenizer
                    except UnboundLocalError:
                        pass
                    clear_memory()
                    model_already_initialized = False

            if "train" in itr_scores:
                scores["train"].append(itr_scores["train"])
            scores["test"].append(itr_scores["test"])

        all_scores = log_scores(
            dataset_name=self.dataset_config.pretty_name,
            metric_configs=self.dataset_config.task.metrics,
            scores=scores,
            model_id=model_config.model_id,
        )

        return all_scores, metadata_dict

    def _get_metadata(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
    ) -> Dict[str, int]:

        # Store the number of parameters in the model, the maximum sequence length and
        # the size of the model's vocabulary
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        max_seq_length = tokenizer.model_max_length
        if hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        elif hasattr(tokenizer, "vocab_size"):
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = -1

        # Store the metadata in a dictionary
        metadata_dict = dict(
            num_model_parameters=num_params,
            max_sequence_length=max_seq_length,
            vocabulary_size=vocab_size,
        )

        # Log the metadata
        logger.info(
            f"The model has {num_params:,} parameters, a vocabulary size of "
            f"{vocab_size:,} and a maximum sequence length of {max_seq_length:,}."
        )

        return metadata_dict

    def _get_training_args(self, iteration_idx: int) -> TrainingArguments:

        # Set the logging strategy
        if self.benchmark_config.verbose:
            logging_strategy = IntervalStrategy.STEPS
        else:
            logging_strategy = IntervalStrategy.NO

        # Set seed variable
        seed = 4242 + iteration_idx

        # Initialise training arguments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            training_args = TrainingArguments(
                output_dir=self.benchmark_config.cache_dir,
                evaluation_strategy=IntervalStrategy.STEPS,
                logging_strategy=logging_strategy,
                save_strategy=IntervalStrategy.STEPS,
                eval_steps=30,
                logging_steps=30,
                save_steps=30,
                max_steps=10_000 if not self.benchmark_config.testing else 10,
                report_to=[],
                save_total_limit=1,
                per_device_train_batch_size=self.benchmark_config.batch_size,
                per_device_eval_batch_size=self.benchmark_config.batch_size,
                learning_rate=2e-5,
                warmup_ratio=0.01,
                gradient_accumulation_steps=1,
                load_best_model_at_end=True,
                optim=OptimizerNames.ADAMW_TORCH,
                seed=seed,
                use_mps_device=torch.backends.mps.is_available(),
                fp16=False,
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
        iteration_idx: int,
        model_config: ModelConfig,
        train: Dataset,
        prepared_train: Dataset,
        prepared_val: Dataset,
        test: Dataset,
        prepared_test: Dataset,
        data_collator: DataCollator,
        training_args: TrainingArguments,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None,
    ) -> Union[Dict[str, Dict[str, float]], Exception]:
        """Run a single iteration of a benchmark.

        Args:
            iteration_idx (int):
                The index of the iteration.
            model_config (ModelConfig):
                The model configuration.
            train (Dataset):
                The original training dataset.
            prepared_train (Dataset):
                The prepared training dataset.
            prepared_val (Dataset):
                The prepared validation dataset.
            test (Dataset):
                The original test dataset.
            prepared_test (Dataset):
                The prepared test dataset.
            data_collator (DataCollator):
                The data collator.
            training_args (TrainingArguments):
                The training arguments.
            tokenizer (PreTrainedTokenizer or None, optional):
                The tokenizer to use in the benchmark. If None then a new tokenizer
                will be loaded. Defaults to None.
            model (PreTrainedModel or None, optional):
                The model to use in the benchmark. If None then a new model will be
                loaded. Defaults to None.

        Returns:
            dict or Exception:
                A dictionary containing the scores for the current iteration, with keys
                `train` and `test`. If an exception is raised, then the exception is
                returned.
        """
        scores: Dict[str, Dict[str, float]] = dict()
        try:
            # Set random seeds to enforce reproducibility of the randomly initialised
            # weights
            seed = 4242 + iteration_idx
            enforce_reproducibility(framework=model_config.framework, seed=seed)

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

            # Initialise trainer
            trainer = self._get_trainer(
                model=model,
                args=training_args,
                train_dataset=prepared_train,
                eval_dataset=prepared_val,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping],
            )

            # Remove trainer logging if not in verbose mode
            if not self.benchmark_config.verbose:
                trainer.log = lambda logs: None

            # Re-block terminal output, as it gets unblocked by the `transformers`
            # package before training
            block_terminal_output()

            # Remove the callback which prints the scores after each evaluation
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
                train_scores = self._evaluate_dataset(
                    dataset=train,
                    prepared_dataset=prepared_train,
                    metric_key_prefix="train",
                    trainer=trainer,
                )
                scores["train"] = train_scores

            # Log test scores
            test_scores = self._evaluate_dataset(
                dataset=test,
                prepared_dataset=prepared_test,
                metric_key_prefix="test",
                trainer=trainer,
            )
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
        """Get a Trainer object.

        Args:
            model (PreTrainedModel):
                The model to finetune.
            args (TrainingArguments):
                The training arguments.
            train_dataset (Dataset):
                The training dataset.
            eval_dataset (Dataset):
                The evaluation dataset.
            tokenizer (PreTrainedTokenizer):
                The tokenizer.
            data_collator (DataCollator):
                The data collator.
            compute_metrics (Callable):
                The function used to compute the metrics.
            callbacks (list of TrainerCallback):
                The callbacks to use.

        Returns:
            Trainer:
                The Trainer object.
        """
        return Trainer(
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
        trainer: Trainer,
        dataset: Dataset,
        prepared_dataset: Dataset,
        metric_key_prefix: str,
    ) -> Dict[str, float]:
        """Evaluate a dataset.

        Args:
            trainer (Trainer):
                The trainer.
            dataset (Dataset):
                The original dataset.
            prepared_dataset (Dataset):
                The prepared dataset.
            metric_key_prefix (str):
                The prefix to use for the metric keys.

        Returns:
            dict:
                The scores.
        """
        return trainer.evaluate(
            eval_dataset=prepared_dataset, metric_key_prefix=metric_key_prefix
        )

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
        self, tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> DataCollator:
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (PreTrainedTokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        pass

    def _compute_metrics(
        self,
        model_outputs_and_labels: Tuple[Sequence, Sequence],
        id2label: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            model_outputs_and_labels (pair of sequences):
                The first sequence contains the model outputs and the second sequence
                contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

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
