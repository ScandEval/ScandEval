"""Functions related to the finetuning of models."""

import logging
import warnings
from collections import defaultdict
from functools import partial
from typing import Callable

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    IntervalStrategy,
    PreTrainedModel,
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer import OptimizerNames

from .callbacks import NeverLeaveProgressCallback
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .model_loading import load_model
from .model_setups import Tokenizer
from .utils import (
    block_terminal_output,
    clear_memory,
    enforce_reproducibility,
    handle_error,
)

logger = logging.getLogger(__package__)


def finetune(
    itr: tqdm,
    train: Dataset,
    val: Dataset,
    tests: list[Dataset],
    prepared_train: Dataset,
    prepared_val: Dataset,
    prepared_tests: list[Dataset],
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    compute_metrics: Callable,
    data_collator: DataCollator,
) -> dict[str, list[dict[str, float]]]:
    """Evaluate a model on a dataset through finetuning.

    Args:
        itr:
            The progress bar iterator.
        train:
            The training dataset.
        val:
            The validation dataset.
        tests:
            The bootstrapped test datasets.
        prepared_train:
            The prepared training dataset.
        prepared_val:
            The prepared validation dataset.
        prepared_tests:
            The prepared bootstrapped test datasets.
        model:
            The model to evaluate.
        tokenizer:
            The tokenizer to use.
        model_config:
            The configuration of the model.
        benchmark_config:
            The benchmark configuration.
        dataset_config:
            The dataset configuration.
        compute_metrics:
            The function used to compute the metrics.
        data_collator:
            The data collator to use.

    Returns:
        A dictionary containing the scores, with keys "test" and maybe "train", with
        values being lists of dicts containing the scores for each metric for each
        iteration.
    """
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)

    bs: int = benchmark_config.batch_size
    ga: int = 32 // bs
    for idx in itr:
        # Set variable that tracks whether we need to initialize new models in
        # the `finetune_single_iteration` call
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
            test = tests[idx]
            prepared_test = prepared_tests[idx]
            assert isinstance(test, Dataset)
            assert isinstance(prepared_test, Dataset)

            # Re-block terminal output, as it gets unblocked by the
            # `transformers` package before training
            block_terminal_output()

            training_args = get_training_args(
                benchmark_config=benchmark_config,
                model_config=model_config,
                iteration_idx=idx,
            )

            # Set the correct batch size and gradient accumulation
            training_args.per_device_train_batch_size = bs
            training_args.per_device_eval_batch_size = bs
            training_args.gradient_accumulation_steps = ga

            itr_scores = finetune_single_iteration(
                iteration_idx=idx,
                model_config=model_config,
                train=train,
                prepared_train=prepared_train,
                prepared_val=prepared_val,
                test=test,
                prepared_test=prepared_test,
                training_args=training_args,
                benchmark_config=benchmark_config,
                dataset_config=dataset_config,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer if model_already_initialized else None,
                model=model if model_already_initialized else None,
            )

            # If the iteration was successful then break the loop
            if isinstance(itr_scores, dict):
                break

            # Otherwise we encountered an error, so we have to deal with it and
            # try again
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
            logger.debug(f"Train scores for iteration {idx}: {itr_scores['train']}")
            scores["train"].append(itr_scores["train"])
        scores["test"].append(itr_scores["test"])
        logger.debug(f"Test scores for iteration {idx}: {itr_scores['test']}")

    return scores


def finetune_single_iteration(
    iteration_idx: int,
    model_config: ModelConfig,
    train: Dataset,
    test: Dataset,
    prepared_train: Dataset,
    prepared_val: Dataset,
    prepared_test: Dataset,
    training_args: TrainingArguments,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    data_collator: DataCollator,
    compute_metrics: Callable,
    tokenizer: Tokenizer | None,
    model: PreTrainedModel | None,
) -> dict[str, dict[str, float]] | Exception:
    """Run a single iteration of a benchmark.

    Args:
        iteration_idx:
            The index of the iteration.
        model_config:
            The model configuration.
        train:
            The original training dataset.
        test:
            The original test dataset.
        prepared_train:
            The prepared training dataset.
        prepared_val:
            The prepared validation dataset.
        prepared_test:
            The prepared test dataset.
        training_args:
            The training arguments.
        benchmark_config:
            The benchmark configuration.
        dataset_config:
            The dataset configuration.
        data_collator:
            The data collator.
        compute_metrics:
            The function to compute the metrics.
        tokenizer:
            The tokenizer to use in the benchmark. If None then a new tokenizer
            will be loaded.
        model:
            The model to use in the benchmark. If None then a new model will be
            loaded.

    Returns:
        A dictionary containing the scores for the current iteration, with keys `train`
        and `test`. If an exception is raised, then the exception is returned.
    """
    scores: dict[str, dict[str, float]] = dict()
    try:
        # Set random seeds to enforce reproducibility of the randomly initialised
        # weights
        seed = 4242 + iteration_idx
        enforce_reproducibility(framework=model_config.framework, seed=seed)

        if tokenizer is None or model is None:
            tokenizer, model_or_generative_model = load_model(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )
            assert isinstance(model_or_generative_model, PreTrainedModel)
            model = model_or_generative_model

        # Initialise compute_metrics function
        compute_metrics = partial(compute_metrics, id2label=dataset_config.id2label)

        # Initialise early stopping callback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

        # Initialise trainer
        trainer = get_trainer(
            model=model,
            args=training_args,
            train_dataset=prepared_train,
            eval_dataset=prepared_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
            data_collator=data_collator,
        )

        if not benchmark_config.verbose:

            def no_logging(logs: dict[str, float]) -> None:
                return

            trainer.log = no_logging

        # Re-block terminal output, as it gets unblocked by the `transformers`
        # package before training
        block_terminal_output()

        # Sort out callbacks. We remove the callbacks that are producing unnecessary
        # output, to avoid cluttering the terminal output
        if not benchmark_config.verbose:
            trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        if benchmark_config.progress_bar:
            trainer.add_callback(NeverLeaveProgressCallback)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            trainer.train()

        if benchmark_config.evaluate_train:
            with torch.inference_mode():
                train_scores = trainer.evaluate(
                    eval_dataset=prepared_train, metric_key_prefix="train"
                )
            scores["train"] = train_scores

        with torch.inference_mode():
            test_scores = trainer.evaluate(
                eval_dataset=prepared_test, metric_key_prefix="test"
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


def get_training_args(
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
    iteration_idx: int,
) -> TrainingArguments:
    """Get the training arguments for the current iteration.

    Args:
        benchmark_config:
            The benchmark configuration.
        model_config:
            The model configuration.
        iteration_idx:
            The index of the current iteration. This is only used to generate a
            unique random seed for the current iteration.

    Returns:
        The training arguments for the current iteration.
    """
    # Set the logging strategy
    if benchmark_config.verbose:
        logging_strategy = IntervalStrategy.STEPS
    else:
        logging_strategy = IntervalStrategy.NO

    # Set seed variable
    seed = 4242 + iteration_idx

    # Initialise training arguments
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        training_args = TrainingArguments(
            output_dir=model_config.model_cache_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
            logging_strategy=logging_strategy,
            save_strategy=IntervalStrategy.STEPS,
            eval_steps=30,
            logging_steps=30,
            save_steps=30,
            max_steps=10_000 if not benchmark_config.testing else 10,
            report_to=[],
            save_total_limit=1,
            per_device_train_batch_size=benchmark_config.batch_size,
            per_device_eval_batch_size=benchmark_config.batch_size,
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
    if benchmark_config.progress_bar:
        training_args.disable_tqdm = False

    return training_args


def get_trainer(
    model: PreTrainedModel,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Tokenizer,
    compute_metrics: Callable,
    data_collator: DataCollator,
    callbacks: list[TrainerCallback],
) -> Trainer:
    """Get a Trainer object.

    Args:
        model:
            The model to finetune.
        args:
            The training arguments.
        train_dataset:
            The training dataset.
        eval_dataset:
            The evaluation dataset.
        tokenizer:
            The tokenizer.
        compute_metrics:
            The function used to compute the metrics.
        data_collator:
            The data collator.
        callbacks:
            The callbacks to use.

    Returns:
        The Trainer object.
    """
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        data_collator=data_collator,
    )
