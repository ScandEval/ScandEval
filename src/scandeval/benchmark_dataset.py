"""Abstract benchmarking dataset class."""

import logging
import random
import subprocess
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import transformers.utils.logging as tf_logging
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from torch.nn.parameter import Parameter
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollator,
    EarlyStoppingCallback,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProgressCallback,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .callbacks import NeverLeaveProgressCallback
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .hf_hub import get_model_config
from .scores import log_scores
from .training_args_with_mps_support import TrainingArgumentsWithMPSSupport
from .utils import clear_memory, enforce_reproducibility, is_module_installed

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
    ):
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
    def benchmark(self, model_id: str) -> Dict[str, dict]:
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
        model_dict = self._load_model(model_config=model_config)

        # Load the dataset dictinoary
        dataset_dict = self._load_data()

        # Process the datasets
        dataset_dict = self._process_data(dataset_dict)

        # Extract the dataset splits
        train = dataset_dict["train"]
        val = dataset_dict["val"]
        test = dataset_dict["test"]

        # Remove empty examples from the datasets
        try:
            train = train.filter(lambda x: len(x["tokens"]) > 0)
            val = val.filter(lambda x: len(x["tokens"]) > 0)
            test = test.filter(lambda x: len(x["tokens"]) > 0)
        except KeyError:
            try:
                train = train.filter(lambda x: len(x["doc"]) > 0)
                val = val.filter(lambda x: len(x["doc"]) > 0)
                test = test.filter(lambda x: len(x["doc"]) > 0)
            except KeyError:
                pass

        # Set variable with number of iterations
        num_iter = 10 if not self.benchmark_config.testing else 2

        if model_config.framework in {"pytorch", "jax"}:
            return self._benchmark_pytorch_jax(
                model_dict=model_dict,
                model_config=model_config,
                train=train,
                val=val,
                test=test,
                rng=rng,
                num_iter=num_iter,
            )

        elif model_config.framework == "spacy":
            return self._benchmark_spacy(
                model_dict=model_dict,
                model_config=model_config,
                train=train,
                test=test,
                rng=rng,
                num_iter=num_iter,
            )

        else:
            raise RuntimeError(
                f'The framework "{model_config.framework}" is not supported!'
            )

    def _benchmark_pytorch_jax(
        self,
        model_dict: dict,
        model_config: ModelConfig,
        train: Dataset,
        val: Dataset,
        test: Dataset,
        rng: np.random.Generator,
        num_iter: int,
    ) -> Dict[str, dict]:
        """Benchmark a PyTorch or JAX model.

        Args:
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            model_config (ModelConfig):
                The model configuration.
            train (Dataset):
                The training dataset.
            val (Dataset):
                The validation dataset.
            test (Dataset):
                The test dataset.
            rng (np.random.Generator):
                The random number generator, used to generate bootstrapped versions of
                the test dataset.
            num_iter (int):
                The number of bootstrapped samples of the test dataset to use.

        Returns:
            dict:
                The keys in the dict are 'raw' and 'total', with all the raw scores in
                the first dictionary and the aggregated scores in the second.
        """
        # Define variable that determines if the model should be finetuned
        finetune = model_config.task == "fill-mask"

        # Deprecation warning if we are not finetuning
        if not finetune:
            logger.warning(
                "Note that support for evaluation of all finetuned models is being "
                "phased out in ScandEval. This is because many of these models have "
                "been trained on part of the ScandEval test sets and the evaluation "
                "scores will thus be artificially large. "
                "To do this properly you can check out our new package `aiai`, which "
                "focuses on evaluation of finetuned models, among other things - "
                "note that this is currently under development, however."
            )

        # Extract the model and tokenizer
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        # Log the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of model parameters: {num_params:,}")

        # Preprocess the datasets
        try:
            params = dict(framework="pytorch", config=model.config, tokenizer=tokenizer)
            if finetune or self.benchmark_config.evaluate_train:
                train = self._preprocess_data(train, **params)
                val = self._preprocess_data(val, **params)
            test = self._preprocess_data(test, **params)
        except ValueError:
            raise InvalidBenchmark("Preprocessing of the dataset could not be done.")

        # If we are testing then truncate the test set
        if self.benchmark_config.testing:
            test = test.select(range(128))

        # Get bootstrap sample indices
        test_bidxs = rng.integers(0, len(test), size=(num_iter, len(test)))

        # Get bootstrapped datasets
        tests = [
            Dataset.from_dict(test[test_bidxs[idx]])
            for idx in range(test_bidxs.shape[0])
        ]

        # Set up progress bar
        if finetune:
            if self.benchmark_config.progress_bar:
                itr = tqdm(range(num_iter), desc="Benchmarking")
            else:
                itr = range(num_iter)
        else:
            itr = [0]

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        # Initialise training arguments
        training_args = TrainingArgumentsWithMPSSupport(
            output_dir=self.benchmark_config.cache_dir,
            evaluation_strategy="steps",
            logging_strategy="steps" if self.benchmark_config.verbose else "no",
            save_strategy="steps",
            eval_steps=30,
            logging_steps=30,
            save_steps=30,
            max_steps=10_000 if not self.benchmark_config.testing else 2,
            report_to="none",
            save_total_limit=1,
            per_device_train_batch_size=32 if not self.benchmark_config.testing else 1,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.01,
            gradient_accumulation_steps=1,
            load_best_model_at_end=True,
            optim="adamw_torch",
            seed=4242,
            bf16=torch.cuda.is_available() or not torch.backends.mps.is_available(),
            no_cuda=self.benchmark_config.testing,
        )

        # Manually set `disable_tqdm` to `False` if `progress_bar` is `True`
        if self.benchmark_config.progress_bar:
            training_args.disable_tqdm = False

        scores = defaultdict(list)
        for idx in itr:
            while True:
                itr_scores = self._benchmark_pytorch_jax_single_iteration(
                    idx=idx,
                    model_config=model_config,
                    train=train,
                    val=val,
                    tests=tests,
                    data_collator=data_collator,
                    finetune=finetune,
                    training_args=training_args,
                )

                # If the iteration was successful then break the loop
                if isinstance(itr_scores, dict):
                    break

                # Otherwise we encountered an error, so we have to deal with it and try
                # again
                else:
                    bs, ga = self._handle_error(
                        e=itr_scores, training_args=training_args
                    )

                    training_args.per_device_train_batch_size = bs // 2
                    training_args.per_device_eval_batch_size = bs // 2
                    training_args.gradient_accumulation_steps = ga * 2

            if "train" in itr_scores:
                scores["train"].append(itr_scores["train"])
            scores["test"].append(itr_scores["test"])

        all_scores = log_scores(
            dataset_name=self.dataset_config.pretty_name,
            metric_configs=self.dataset_config.task.metrics,
            scores=scores,
            model_id=model_config.model_id,
            finetuned=finetune,
        )

        # Garbage collection, to avoid memory issues
        try:
            del model
        except UnboundLocalError:
            pass
        try:
            del model_dict
        except UnboundLocalError:
            pass
        clear_memory()

        return all_scores

    def _handle_error(
        self, e: Exception, training_args: TrainingArguments
    ) -> Tuple[int, int]:
        """Handle an error that occurred during the benchmarking process.

        Args:
            e (Exception):
                The exception that was raised.
            training_args (TrainingArguments):
                The training arguments that were used.

        Returns:
            pair of int:
                The batch size and gradient accumulation steps to use.
        """
        # We assume that all these CUDA errors are caused by
        # insufficient GPU memory
        # TODO: Handle MPS out of memory as well
        cuda_errs = ["CUDA out of memory", "CUDA error"]

        # If it is an unknown error, then simply report it
        if all([err not in str(e) for err in cuda_errs]):
            raise InvalidBenchmark(str(e))

        # If it is a CUDA memory error, then reduce batch size and up
        # gradient accumulation
        bs = training_args.per_device_train_batch_size
        ga = training_args.gradient_accumulation_steps
        if bs == 1:
            raise InvalidBenchmark("CUDA out of memory, even with a batch size of 1!")
        return bs // 2, ga * 2

    def _benchmark_pytorch_jax_single_iteration(
        self,
        idx: int,
        model_config: ModelConfig,
        train: Dataset,
        val: Dataset,
        tests: Sequence[Dataset],
        data_collator: DataCollator,
        finetune: bool,
        training_args: TrainingArguments,
    ) -> Union[dict, Exception]:
        """Run a single iteration of a PyTorch/JAX benchmark.

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
            finetune (bool):
                Whether the model is being finetuned.
            training_args (TrainingArguments):
                The training arguments.

        Returns:
            dict or Exception:
                A dictionary containing the scores for the current iteration, with keys
                `train` and `test`. If an exception is raised, then the exception is
                returned.
        """
        scores = dict()
        try:
            # Set random seeds to enforce reproducibility of the randomly
            # initialised weights
            training_args.seed = 4242 + idx
            random.seed(4242 + idx)
            np.random.seed(4242 + idx)
            torch.manual_seed(4242 + idx)
            torch.cuda.manual_seed_all(4242 + idx)

            # Reinitialise a new model
            model_dict = self._load_model(model_config=model_config)
            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]

            # Initialise compute_metrics function
            compute_metrics = partial(
                self._compute_metrics, id2label=model.config.id2label
            )

            # Initialise early stopping callback
            early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

            # Disable logging from trainer.py
            (logging.getLogger("transformers.trainer").setLevel(logging.ERROR))

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
            tf_logging.set_verbosity_error()

            # Remove trainer logging
            trainer.log = lambda _: None

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
            if finetune:
                trainer.train()

            # Log training scores and save the state
            if self.benchmark_config.evaluate_train:
                train_scores = trainer.evaluate(train, metric_key_prefix="train")
                scores["train"] = train_scores

            # Set up a progress bar for the test datasets if we are not
            # finetuning
            if not finetune:
                if self.benchmark_config.progress_bar:
                    test_itr = tqdm(tests, desc="Benchmarking")
                else:
                    test_itr = tests
            else:
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
                del model_dict
            except UnboundLocalError:
                pass
            clear_memory()
            return e

    def _benchmark_spacy(
        self,
        model_dict: dict,
        model_config: ModelConfig,
        train: Dataset,
        test: Dataset,
        rng: np.random.Generator,
        num_iter: int,
    ) -> Dict[str, dict]:
        """Benchmark a spaCy model.

        Args:
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            model_config (ModelConfig):
                The model configuration.
            train (Dataset):
                The training dataset.
            test (Dataset):
                The test dataset.
            rng (np.random.Generator):
                The random number generator, used to generate bootstrapped versions of
                the test dataset.
            num_iter (int):
                The number of bootstrapped samples of the test dataset to use.

        Returns:
            dict:
                The keys in the dict are 'raw' and 'total', with all the raw scores in
                the first dictionary and the aggregated scores in the second.
        """
        # Get bootstrap sample indices
        test_bidxs = rng.integers(0, len(test), size=(num_iter, len(test)))

        # Load the model
        model = model_dict["model"]

        # Preprocess the test datasets
        test = self._preprocess_data(test, framework="spacy")
        tests = [
            Dataset.from_dict(test[test_bidxs[idx]])
            for idx in range(test_bidxs.shape[0])
        ]

        # Get the test predictions
        all_test_scores = list()
        for dataset in tqdm(tests, desc="Benchmarking"):
            preds_labels = self._get_spacy_predictions_and_labels(
                model=model,
                dataset=dataset,
            )

            # Check if the spaCy model has been trained on the task at hand. If
            # not, then skip this benchmark.
            sample_preds = preds_labels[0][0]
            pos_ner_test = isinstance(sample_preds, list) and "" in sample_preds
            dep_test = isinstance(sample_preds[0], list) and "" in sample_preds[0]
            if pos_ner_test or dep_test:
                raise InvalidBenchmark(
                    "This spaCy model have not been trained on this task. Skipping."
                )

            test_scores = self._compute_metrics(preds_labels)
            test_scores = {f"test_{key}": val for key, val in test_scores.items()}
            all_test_scores.append(test_scores)
        scores = dict(test=all_test_scores)

        if self.benchmark_config.evaluate_train:

            # Preprocess the train datasets
            train = self._preprocess_data(train, framework=model_config.framework)

            # Get the train predictions
            preds_labels = self._get_spacy_predictions_and_labels(
                model=model,
                dataset=train,
            )

            # Compute the train scores
            train_scores = self._compute_metrics(preds_labels)
            train_scores = {f"train_{key}": val for key, val in train_scores.items()}

            # Store the train scores
            scores["train"] = [train_scores]

        # Log the scores
        all_scores = log_scores(
            dataset_name=self.dataset_config.name,
            metric_configs=self.dataset_config.task.metrics,
            scores=scores,
            model_id=model_config.model_id,
            finetuned=False,
        )

        # Garbage collection, to avoid memory issues
        try:
            del model
        except UnboundLocalError:
            pass
        try:
            del model_dict
        except UnboundLocalError:
            pass
        clear_memory()

        return all_scores

    def __call__(self, *args, **kwargs):
        return self.benchmark(*args, **kwargs)

    # TODO: Cache this
    def _load_data(self) -> DatasetDict:
        """Load the datasets.

        Returns:
            DatasetDict:
                A dictionary containing the 'train', 'val' and 'test' splits of the
                dataset.
        """
        # Download dataset from the HF Hub
        dataset_dict = load_dataset(
            path=self.dataset_config.huggingface_id,
            use_auth_token=self.benchmark_config.use_auth_token,
            cache_dir=self.benchmark_config.cache_dir,
        )

        # Remove all other keys than 'train', 'val' and 'test'
        dataset_dict = {key: dataset_dict[key] for key in ["train", "val", "test"]}

        # Return the dataset dictionary
        return dataset_dict

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

    # TODO: Cache this
    def _load_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Load the model.

        Args:
            model_config (ModelConfig):
                The model configuration.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the value being
                the model. Can contain other objects related to the model, such as its
                tokenizer.

        Raises:
            RuntimeError: If the framework is not recognized.
        """
        # Ensure that the framework is installed
        from_flax = model_config.framework == "jax"
        try:
            # If the framework is JAX then change it to PyTorch, since we will convert
            # JAX models to PyTorch upon download
            if model_config.framework == "jax":
                model_config.framework = "pytorch"

            elif model_config.framework == "spacy":
                import spacy

                # Ignore warnings from spaCy. This has to be called after the import,
                # as the __init__.py file of spaCy sets the warning levels of spaCy
                # warning W036
                warnings.filterwarnings("ignore", module="spacy*")

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"The model {model_config.model_id} is built using the spaCy "
                "framework which is not installed. ScandEval is phasing out support "
                "for this framework and will instead focus on evaluating pretrained "
                "language models. You can still evaluate this spaCy model on the "
                "task if you install spaCy first (`pip install spacy`), but keep in "
                "mind that the results will be artificially large, as the spaCy "
                "models have been trained on part of the ScandEval test sets (which "
                "is exactly the reason why we are phasing out support for evaluating "
                "finetuned models in general. To do this properly you can check out "
                "our new package `aiai`, which focuses on evaluation of finetuned "
                "models, among other things - note that this is currently under "
                "development, however."
            )

        if model_config.framework == "pytorch":
            return self._load_pytorch_model(model_config, from_flax=from_flax)

        elif model_config.framework == "spacy":
            logger.warning(
                "Note that support for spaCy models (and generally evaluation of all "
                "finetuned models) is being phased out in ScandEval. This is because "
                "the spaCy models have been trained on part of the ScandEval test "
                "sets and the evaluation scores will thus be artificially large. "
                "To do this properly you can check out our new package `aiai`, which "
                "focuses on evaluation of finetuned models, among other things - "
                "note that this is currently under development, however."
            )
            return self._load_spacy_model(model_config)

        else:
            raise RuntimeError(
                f'The framework "{model_config.framework}" is not supported!'
            )

    def _load_pytorch_model(
        self,
        model_config: ModelConfig,
        from_flax: bool,
    ) -> Dict[str, Any]:
        """Load a PyTorch model.

        Args:
            model_config (ModelConfig):
                The configuration of the model.
            from_flax (bool):
                Whether the model is a Flax model.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the value being
                the model. Can contain other objects related to the model, such as its
                tokenizer.
        """
        # Set the parameters if we are finetuning
        params = dict()
        if model_config.task == "fill-mask":
            params = dict(
                num_labels=self.dataset_config.num_labels,
                id2label=self.dataset_config.id2label,
                label2id=self.dataset_config.label2id,
            )

        try:
            # If the model ID specifies a random model, then load that.
            if model_config.model_id.startswith("random"):
                if model_config.model_id == "random-xlmr-base-sequence-clf":
                    rnd_model = "xlm-roberta-base"
                    model_cls = RobertaForSequenceClassification
                elif model_config.model_id == "random-xlmr-base-token-clf":
                    rnd_model = "xlm-roberta-base"
                    model_cls = RobertaForTokenClassification
                elif model_config.model_id == "random-electra-small-sequence-clf":
                    rnd_model = "google/electra-small-discriminator"
                    model_cls = ElectraForSequenceClassification
                elif model_config.model_id == "random-electra-small-token-clf":
                    rnd_model = "google/electra-small-discriminator"
                    model_cls = ElectraForTokenClassification
                else:
                    raise ValueError(
                        f"A random model was chosen, `{model_config.model_id}`, "
                        "but it was not recognized."
                    )

                config = AutoConfig.from_pretrained(
                    rnd_model,
                    use_auth_token=self.benchmark_config.use_auth_token,
                    **params,
                )
                model = model_cls(config)

            # Otherwise load the pretrained model
            else:
                config = AutoConfig.from_pretrained(
                    model_config.model_id,
                    revision=model_config.revision,
                    use_auth_token=self.benchmark_config.use_auth_token,
                    **params,
                )

                supertask = self.dataset_config.task.supertask
                if supertask == "token-classification":
                    model_cls = AutoModelForTokenClassification
                elif supertask == "text-classification":
                    model_cls = AutoModelForSequenceClassification
                elif supertask == "question-answering":
                    model_cls = AutoModelForQuestionAnswering
                else:
                    raise ValueError(f"The supertask `{supertask}` was not recognised.")

                model = model_cls.from_pretrained(
                    model_config.model_id,
                    revision=model_config.revision,
                    use_auth_token=self.benchmark_config.use_auth_token,
                    config=config,
                    cache_dir=self.benchmark_config.cache_dir,
                    from_flax=from_flax,
                )

        except (OSError, ValueError):
            msg = (
                f"The model {model_config.model_id} either does not exist on the "
                "Hugging Face Hub, or it has no frameworks registered, or it is a "
                "private model. If it *does* exist on the Hub and is a public "
                "model then please ensure that it has a framework registered. If "
                "it is a private model then enable the `--use-auth-token` flag "
                "and make sure that you are logged in to the Hub via the "
                "`huggingface-cli login` command."
            )
            raise InvalidBenchmark(msg)

        # Ensure that the labels of the model are consistent with the labels of the
        # dataset
        self._adjust_label_ids(model=model, model_config=model_config)

        # If the model is a subclass of a RoBERTa model then we have to add a prefix
        # space to the tokens, by the way the model is constructed.
        if model_config.model_id.startswith("random"):
            m_id = rnd_model
        else:
            m_id = model_config.model_id
        prefix = "Roberta" in type(model).__name__
        params = dict(use_fast=True, add_prefix_space=prefix)
        tokenizer = AutoTokenizer.from_pretrained(
            m_id,
            revision=model_config.revision,
            use_auth_token=self.benchmark_config.use_auth_token,
            **params,
        )

        # Set the maximal length of the tokenizer to the model's maximal length.
        # This is required for proper truncation
        if (
            not hasattr(tokenizer, "model_max_length")
            or tokenizer.model_max_length > 1_000
        ):

            if hasattr(tokenizer, "max_model_input_sizes"):
                all_max_lengths = tokenizer.max_model_input_sizes.values()
                if len(list(all_max_lengths)) > 0:
                    min_max_length = min(list(all_max_lengths))
                    tokenizer.model_max_length = min_max_length
                else:
                    tokenizer.model_max_length = 512
            else:
                tokenizer.model_max_length = 512

        return dict(model=model, tokenizer=tokenizer)

    def _load_spacy_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Load a spaCy model.

        Args:
            model_config (ModelConfig):
                The configuration of the model.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the value being
                the model. Can contain other objects related to the model, such as its
                tokenizer.
        """
        import spacy

        # Ignore warnings from spaCy. This has to be called after the import, as the
        # __init__.py file of spaCy sets the warning levels of spaCy warning W036
        warnings.filterwarnings("ignore", module="spacy*")

        local_model_id = model_config.model_id.split("/")[-1]

        # Download the model if it has not already been so
        if not is_module_installed(local_model_id):
            url = (
                f"https://huggingface.co/{model_config.model_id}/resolve/main/"
                f"{local_model_id}-any-py3-none-any.whl"
            )
            subprocess.run(["pip3", "install", url])

        # Load the model
        try:
            model = spacy.load(local_model_id)
        except OSError:
            raise InvalidBenchmark(
                f"The model {model_config.model_id} could not be installed from spaCy."
            )
        return dict(model=model)

    def _adjust_label_ids(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> nn.Module:
        """Adjust the label ids of the model to match the dataset.

        Args:
            model (PyTorch Model):
                The model to adjust the label ids of.
            model_config (ModelConfig):
                The model configuration.

        Returns:
            PyTorch Model:
                The model with adjusted label ids.
        """
        # Define the types of the label conversions
        model_label2id: Optional[dict]
        model_id2label: Optional[Union[dict, list]]

        # Get the `label2id` and `id2label` conversions from the model config
        try:
            model_label2id = dict(model.config.label2id)
        except AttributeError:
            model_label2id = None
        try:
            try:
                model_num_labels = len(model.config.id2label)
                if not isinstance(model.config.id2label, list):
                    model_id2label = dict(model.config.id2label)
                else:
                    model_id2label = model.config.id2label
                model_id2label = [
                    model_id2label[idx].upper() for idx in range(model_num_labels)
                ]
            except IndexError:
                raise InvalidBenchmark(
                    "There is a gap in the indexing dictionary of the model."
                )
        except AttributeError:
            model_id2label = None

        # If one of `label2id` or `id2label` exists in the model config, then define
        # the other one from it
        if model_label2id is not None and model_id2label is None:
            model_id2label = {idx: lbl.upper() for lbl, idx in model_label2id.items()}
            model_id2label = [model_id2label[idx] for idx in range(len(model_id2label))]
            model.config.id2label = model_id2label
        if model_label2id is None and model_id2label is not None:
            model_label2id = {lbl.upper(): id for id, lbl in enumerate(model_id2label)}
            model.config.label2id = model_label2id

        # If the model does not have `label2id` or `id2label` conversions, then use the
        # defaults
        if model_config.task == "fill-mask" or (
            model_label2id is None or model_id2label is None
        ):
            model.config.label2id = self.dataset_config.label2id
            model.config.id2label = self.dataset_config.id2label

        # If the model *does* have conversions, then ensure that it can deal with all
        # the labels in the default conversions. This ensures that we can smoothly deal
        # with labels that the model have not been trained on (it will just always get
        # those labels wrong)
        else:

            # Collect the dataset labels and model labels in the `model_id2label`
            # conversion list
            for label in self.dataset_config.id2label:
                syns = [
                    syn
                    for lst in self.dataset_config.label_synonyms
                    for syn in lst
                    if label.upper() in lst
                ]
                if all([syn not in model_id2label for syn in syns]):
                    model_id2label.append(label)

            # Ensure that the model_id2label does not contain duplicates modulo
            # synonyms
            for idx, label in enumerate(model_id2label):
                try:
                    canonical_syn = [
                        syn_lst
                        for syn_lst in self.dataset_config.label_synonyms
                        if label.upper() in syn_lst
                    ][0][-1]
                    model_id2label[idx] = canonical_syn

                # IndexError appears when the label does not appear within the
                # label_synonyms (i.e. that we added it in the previous step). In this
                # case, we just skip the label.
                except IndexError:
                    continue

            # Get the synonyms of all the labels, new ones included
            new_synonyms = list(self.dataset_config.label_synonyms)
            flat_old_synonyms = [
                syn for lst in self.dataset_config.label_synonyms for syn in lst
            ]
            new_synonyms += [
                [label.upper()]
                for label in model_id2label
                if label.upper() not in flat_old_synonyms
            ]

            # Add all the synonyms of the labels into the label2id conversion
            # dictionary
            model_label2id = {
                label.upper(): id
                for id, lbl in enumerate(model_id2label)
                for label_syns in new_synonyms
                for label in label_syns
                if lbl.upper() in label_syns
            }

            # Get the old id2label conversion
            old_id2label = [
                model.config.id2label[idx].upper()
                for idx in range(len(model.config.id2label))
            ]

            # Alter the model's classification layer to match the dataset if the
            # model is missing labels
            if (
                len(model_id2label) > len(old_id2label)
                and model_config.framework == "pytorch"
            ):
                model = self._alter_classification_layer(
                    model=model,
                    model_id2label=model_id2label,
                    old_id2label=old_id2label,
                    flat_old_synonyms=flat_old_synonyms,
                )

            # Update the model's own conversions with the new ones
            model.config.id2label = model_id2label
            model.config.label2id = model_label2id

        return model

    def _alter_classification_layer(
        self,
        model: nn.Module,
        model_id2label: list,
        old_id2label: list,
        flat_old_synonyms: list,
    ) -> nn.Module:
        """Alter the classification layer of the model to match the dataset.

        This changes the classification layer in the finetuned model to be consistent
        with all the labels in the dataset. If the model was previously finetuned on a
        dataset which left out a label, say, then that label will be inserted in the
        model architecture here, but without the model ever predicting it. This will
        allow the model to be benchmarked on such datasets, however.

        Note that this only works on classification tasks. This code needs to be
        rewritten when we add other types of tasks.

        Args:
            model (PyTorch Model):
                The model to alter the classification layer of.
            model_id2label (list):
                The model's label conversion.
            old_id2label (list):
                The old label conversion.
            flat_old_synonyms (list):
                The synonyms of the old labels.

        Returns:
            PyTorch Model:
                The model with an altered classification layer.
        """
        # Count the number of new labels to add to the model
        num_new_labels = len(model_id2label) - len(old_id2label)

        # If *all* the new labels are new and aren't even synonyms of the
        # model's labels, then raise an exception
        if num_new_labels == self.dataset_config.num_labels:
            if len(set(flat_old_synonyms).intersection(old_id2label)) == 0:
                msg = (
                    "The model has not been trained on any of the labels in the "
                    "dataset, or synonyms thereof."
                )
                raise InvalidBenchmark(msg)

        # Load the weights from the model's current classification layer. This handles
        # both the token classification case and the sequence classification case.
        # NOTE: This might need additional cases (or a general solution) when we start
        #       dealing with other tasks.
        try:
            clf_weight = model.classifier.weight.data
        except AttributeError:
            try:
                clf_weight = model.classifier.out_proj.weight.data
            except AttributeError:
                msg = "Model does not seem to be a classification model."
                raise InvalidBenchmark(msg)

        # Create the new weights, which have zeros at all the new entries
        zeros = torch.zeros(num_new_labels, model.config.hidden_size)
        new_clf_weight = torch.cat((clf_weight, zeros), dim=0)
        new_clf_weight = Parameter(new_clf_weight)

        # Create the new classification layer
        new_clf = nn.Linear(model.config.hidden_size, len(model_id2label))

        # Assign the new weights to the new classification layer, and replace the old
        # classification layer with this one
        new_clf.weight = new_clf_weight
        model.classifier = new_clf

        # Update the number of labels the model thinks it has. This is required to
        # avoid exceptions when evaluating
        model.config.num_labels = len(model_id2label)
        model.num_labels = len(model_id2label)

        return model

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
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
        pass

    def _compute_metrics(
        self, predictions_and_labels: tuple, id2label: Optional[list] = None
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions and the second
                array contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)
        results = dict()
        for cfg in self.dataset_config.task.metrics:
            metric = self._metrics[cfg.name]
            score_dict = metric.compute(
                predictions=predictions,
                references=labels,
                **cfg.compute_kwargs,
            )
            if score_dict is not None:
                scores = score_dict[cfg.results_key]
                results[cfg.name] = scores
        return results

    @abstractmethod
    def _get_spacy_predictions_and_labels(self, model, dataset: Dataset) -> tuple:
        """Get predictions from spaCy model on dataset.

        Args:
            model (spaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the second
                array contains the true labels.
        """
        pass
