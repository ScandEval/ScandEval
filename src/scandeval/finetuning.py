"""Functions related to the finetuning of models."""

import importlib.util
import logging
import sys
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Type

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
    TrainingArguments,
)
from transformers.trainer import OptimizerNames

from .callbacks import NeverLeaveProgressCallback
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .enums import DataType
from .exceptions import InvalidBenchmark, NaNValueInModelOutput
from .model_loading import load_model
from .protocols import Tokenizer
from .utils import block_terminal_output, clear_memory, enforce_reproducibility

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
    trainer_class: Type[Trainer],
    evaluate_inputs_fn: Callable[..., dict[str, Any]],
    preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        trainer_class:
            The Trainer class to use.
        evaluate_inputs_fn:
            A function that generates the appropriate inputs for the `Trainer.evaluate`
            method.
        preprocess_logits_for_metrics:
            A function that preprocesses the logits before they are passed to the
            `compute_metrics` function. This helps prevent memory issues during
            evaluation.

    Returns:
        A dictionary containing the scores, with keys "test" and maybe "train", with
        values being lists of dicts containing the scores for each metric for each
        iteration.
    """
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)

    using_cuda = benchmark_config.device == torch.device("cuda")
    if using_cuda and torch.cuda.is_bf16_supported():
        dtype = DataType.BF16
    elif using_cuda:
        dtype = DataType.FP16
    else:
        dtype = DataType.FP32

    bs: int = benchmark_config.batch_size
    for idx in itr:
        # Set variable that tracks whether we need to initialize new models in
        # the single iteration call
        model_already_initialized = idx == 0

        # Run a loop here to deal with automatic reduction of batch size
        while True:
            # Clear GPU memory
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

            try:
                test = tests[idx]
                prepared_test = prepared_tests[idx]
                assert isinstance(test, Dataset)
                assert isinstance(prepared_test, Dataset)

                # Re-block terminal output, as it gets unblocked by the `transformers`
                # package before training
                block_terminal_output()

                training_args = get_training_args(
                    benchmark_config=benchmark_config,
                    model_config=model_config,
                    iteration_idx=idx,
                    dtype=dtype,
                    batch_size=bs,
                )

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
                    trainer_class=trainer_class,
                    evaluate_inputs_fn=evaluate_inputs_fn,
                    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                )

                if "train" in itr_scores:
                    logger.debug(
                        f"Train scores for iteration {idx}: {itr_scores['train']}"
                    )
                    scores["train"].append(itr_scores["train"])
                scores["test"].append(itr_scores["test"])
                logger.debug(f"Test scores for iteration {idx}: {itr_scores['test']}")

                break

            # NaN values can appear in the model output when using mixed precision, as
            # the hidden states get overflowed. In this case we try to disable mixed
            # precision and try again.
            except NaNValueInModelOutput:
                if dtype != DataType.FP32:
                    dtype = DataType.FP32
                    model_already_initialized = False
                    logger.debug(
                        "NaN value detected in model outputs while using mixed "
                        "precision. Retrying with full fp32 precision."
                    )
                else:
                    raise InvalidBenchmark(
                        "NaN value detected in model outputs, even with mixed "
                        "precision disabled."
                    )

            except Exception as e:
                if "CUDA" not in str(e) and "out of memory" not in str(e):
                    raise InvalidBenchmark(str(e))

                if bs <= 1:
                    raise InvalidBenchmark(
                        "Could not benchmark the model, even with a batch size of 1!"
                    )

                model_already_initialized = False

                # Half batch size, and raise error if we've reached 0
                bs //= 2
                logger.debug(f"Reduced batch size to {bs}")

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
    trainer_class: Type[Trainer],
    evaluate_inputs_fn: Callable[..., dict[str, Any]],
    preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, dict[str, float]]:
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
        trainer_class:
            The trainer class to use.
        evaluate_inputs_fn:
            A function that generates the appropriate inputs for the `Trainer.evaluate`
            method.
        preprocess_logits_for_metrics:
            A function that preprocesses the logits before they are passed to the
            `compute_metrics` function. This helps prevent memory issues during
            evaluation.

    Returns:
        A dictionary containing the scores for the current iteration, with keys `train`
        and `test`.
    """
    scores: dict[str, dict[str, float]] = dict()

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

    compute_metrics = partial(compute_metrics, id2label=dataset_config.id2label)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=prepared_train,
        eval_dataset=prepared_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            trainer.train()

        if benchmark_config.evaluate_train:
            with torch.inference_mode():
                evaluate_inputs = evaluate_inputs_fn(
                    dataset=train,
                    prepared_dataset=prepared_train,
                    metric_key_prefix="train",
                )
                train_scores = trainer.evaluate(**evaluate_inputs)
            scores["train"] = train_scores

        with torch.inference_mode():
            evaluate_inputs = evaluate_inputs_fn(
                dataset=test, prepared_dataset=prepared_test, metric_key_prefix="test"
            )
            test_scores = trainer.evaluate(**evaluate_inputs)
        scores["test"] = test_scores

        return scores

    except NaNValueInModelOutput as e:
        del trainer
        del model
        del tokenizer
        clear_memory()
        raise e

    except (RuntimeError, ValueError, IndexError) as e:
        raise InvalidBenchmark(str(e))


def get_training_args(
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
    iteration_idx: int,
    dtype: DataType,
    batch_size: int | None = None,
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
        dtype:
            The data type to use for the model weights.
        batch_size:
            The batch size to use for the current iteration, or None if the batch size
            in the benchmark config should be used.

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

    if batch_size is None:
        batch_size = benchmark_config.batch_size

    if (
        benchmark_config.device == torch.device("cuda")
        and importlib.util.find_spec("bitsandbytes") is not None
    ):
        optimizer = OptimizerNames.ADAMW_8BIT
    else:
        optimizer = OptimizerNames.ADAMW_TORCH

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
            max_steps=2 if hasattr(sys, "_called_from_test") else 10_000,
            use_cpu=benchmark_config.device == torch.device("cpu"),
            report_to=[],
            save_total_limit=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_ratio=0.01,
            gradient_accumulation_steps=32 // batch_size,
            load_best_model_at_end=True,
            optim=optimizer,
            seed=seed,
            fp16=dtype == DataType.FP16,
            bf16=dtype == DataType.BF16,
            disable_tqdm=not benchmark_config.progress_bar,
            ddp_find_unused_parameters=False,
        )

    return training_args
