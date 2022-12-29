"""Benchmarking model inference speed."""

from collections import defaultdict
from typing import Dict, List, Union

import pyinfer
import torch
from tqdm.auto import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from .benchmark_dataset import log_scores
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .model_loading import load_model
from .types import SCORE_DICT


def benchmark_speed(
    itr: tqdm,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> SCORE_DICT:
    """Benchmark model inference speed.

    Args:
        itr (tqdm):
            tqdm iterator.
        tokenizer (PreTrainedTokenizer):
            Tokenizer to use.
        model (PreTrainedModel):
            Model to use.
        model_config (ModelConfig):
            Model configuration.
        dataset_config (DatasetConfig):
            Dataset configuration.
        benchmark_config (BenchmarkConfig):
            Benchmark configuration.

    Returns:
        SCORE_DICT:
            Dictionary of scores.
    """
    # Initialise the `scores` dictionary
    scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for _ in itr:

        # Run the speed benchmark
        itr_scores = benchmark_speed_single_iteration(
            tokenizer=tokenizer,
            model=model,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        # If the iteration was unsuccessful then raise an error
        if isinstance(itr_scores, Exception):
            raise InvalidBenchmark(f"Speed benchmark failed with error: {itr_scores!r}")

        # Otherwise, append the scores to the list and log the result
        else:
            scores["test"].append(itr_scores["test"])
            if benchmark_config.evaluate_train:
                scores["train"].append(itr_scores["train"])
            if benchmark_config.verbose:
                print(itr_scores)

    all_scores = log_scores(
        dataset_name=dataset_config.pretty_name,
        metric_configs=dataset_config.task.metrics,
        scores=scores,
        model_id=model_config.model_id,
    )

    return all_scores


def benchmark_speed_single_iteration(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> Union[Dict[str, Dict[str, float]], Exception]:
    """Run a single iteration of the speed benchmark.

    Args:
        tokenizer (PreTrainedTokenizer):
            The tokenizer to use in the benchmark.
        model (PreTrainedModel):
            The model to use in the benchmark.
        model_config (ModelConfig):
            The model configuration.
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.

    Returns:
        dict or Exception:
            A dictionary containing the scores for the current iteration, with keys
            `train` and `test`. If an exception is raised, then the exception is
            returned.
    """
    scores: Dict[str, Dict[str, float]] = dict()
    try:

        # Reinitialise a new model
        if tokenizer is None or model is None:
            tokenizer, model = load_model(
                model_id=model_config.model_id,
                revision=model_config.revision,
                supertask=dataset_config.task.supertask,
                num_labels=dataset_config.num_labels,
                label2id=dataset_config.label2id,
                id2label=dataset_config.id2label,
                from_flax=model_config.framework == "jax",
                use_auth_token=benchmark_config.use_auth_token,
                cache_dir=benchmark_config.cache_dir,
            )

        # Ensure that the model is on the CPU
        model.cpu()

        # Create a dummy document
        doc = "This is a dummy document. " * 100

        def predict(docs: List[str]) -> None:
            """Function used to benchmark inference speed of the model."""

            # Raise an error if the tokenizer or model is undefined
            if tokenizer is None or model is None:
                raise ValueError("Tokenizer and model must not be None.")

            # Tokenize the document
            inputs = tokenizer(
                docs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Run inference with the model
            with torch.no_grad():
                model(**inputs)

        # Do a warmup run
        pyinfer.InferenceReport(model=predict, inputs=doc, n_iterations=10).run(
            print_report=False
        )

        # Initialise the speed benchmark
        speed_benchmark = pyinfer.InferenceReport(
            model=predict,
            inputs=doc,
            n_iterations=100,
        )

        # Run the speed benchmark
        speed_scores = speed_benchmark.run(print_report=False)

        # Close the speed benchmark
        del speed_benchmark

        # Store the scores
        scores["test"] = {"test_speed": speed_scores["Infer(p/sec)"]}
        if benchmark_config.evaluate_train:
            scores["train"] = {"train_speed": speed_scores["Infer(p/sec)"]}

        # Return the scores
        return scores

    except (RuntimeError, ValueError, IndexError) as e:
        return e
