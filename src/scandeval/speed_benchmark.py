"""Benchmarking model inference speed."""

import logging
from collections import defaultdict

import pyinfer
import torch
from tqdm.auto import tqdm
from transformers.modeling_utils import GenerationConfig, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .model_loading import load_model
from .protocols import GenerativeModel, Tokenizer
from .utils import clear_memory, model_is_generative

logger = logging.getLogger(__package__)


def benchmark_speed(
    itr: tqdm,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> dict[str, list[dict[str, float]]]:
    """Benchmark model inference speed.

    Args:
        itr:
            tqdm iterator.
        tokenizer:
            Tokenizer to use.
        model:
            Model to use.
        model_config:
            Model configuration.
        dataset_config:
            Dataset configuration.
        benchmark_config:
            Benchmark configuration.

    Returns:
        Dictionary of scores.
    """
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)

    for itr_idx in itr:
        itr_scores = benchmark_speed_single_iteration(
            tokenizer=tokenizer,
            model=model,
            itr_idx=itr_idx,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        clear_memory()

        if isinstance(itr_scores, Exception):
            raise InvalidBenchmark(f"Speed benchmark failed with error: {itr_scores!r}")
        else:
            scores["test"].append(itr_scores["test"])
            if benchmark_config.evaluate_train:
                scores["train"].append(itr_scores["train"])
            logger.debug(f"Scores for iteration {itr_idx}: {itr_scores}")

    return scores


def benchmark_speed_single_iteration(
    tokenizer: Tokenizer,
    model: PreTrainedModel | GenerativeModel,
    itr_idx: int,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> dict[str, dict[str, float]] | Exception:
    """Run a single iteration of the speed benchmark.

    Args:
        tokenizer:
            The tokenizer to use in the benchmark.
        model:
            The model to use in the benchmark.
        itr_idx:
            The index of the iteration.
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        A dictionary containing the scores for the current iteration, with keys `train`
        and `test`. If an exception is raised, then the exception is returned.
    """
    is_generative = model_is_generative(model=model)

    scores: dict[str, dict[str, float]] = dict()
    try:
        # Reinitialise a new model
        if tokenizer is None or model is None:
            tokenizer, model = load_model(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )

        def predict(doc: str) -> None:
            """Function used to benchmark inference speed of the model."""
            # Raise an error if the tokenizer or model is undefined
            if tokenizer is None or model is None:
                raise ValueError("Tokenizer and model must not be None.")

            # Tokenize the document
            inputs = tokenizer(doc, padding=True, truncation=True, return_tensors="pt")

            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

            # Run inference with the model
            with torch.inference_mode():
                if is_generative:
                    model.generate(
                        inputs=inputs["input_ids"],
                        generation_config=GenerationConfig(
                            max_new_tokens=1,
                            pad_token_id=model.config.pad_token_id,
                            eos_token_id=model.config.eos_token_id,
                            do_sample=False,
                        ),
                    )
                else:
                    assert isinstance(model, PreTrainedModel)
                    model(**inputs)

        base_doc = "Document which contains roughly 10 tokens. "
        multiplier = 10 * (1 + itr_idx)
        doc = base_doc * multiplier
        short_multiplier = 1.25 * (1 + itr_idx)
        short_doc = base_doc * round(short_multiplier)

        # Do a warmup run, as the first run is always slower
        pyinfer.InferenceReport(model=predict, inputs=base_doc, n_seconds=1).run(
            print_report=False
        )

        speed_scores = pyinfer.InferenceReport(
            model=predict, inputs=doc, n_seconds=3
        ).run(print_report=False)
        num_tokens = len(tokenizer(doc, truncation=True)["input_ids"])
        tokens_per_second = speed_scores["Infer(p/sec)"] * num_tokens

        speed_scores_short = pyinfer.InferenceReport(
            model=predict, inputs=short_doc, n_seconds=3
        ).run(print_report=False)
        num_tokens_short = len(tokenizer(short_doc, truncation=True)["input_ids"])
        tokens_per_second_short = speed_scores_short["Infer(p/sec)"] * num_tokens_short

        scores["test"] = dict(
            test_speed=tokens_per_second, test_speed_short=tokens_per_second_short
        )
        if benchmark_config.evaluate_train:
            scores["train"] = dict(
                train_speed=tokens_per_second, train_speed_short=tokens_per_second_short
            )

        # Return the scores
        return scores

    except (RuntimeError, ValueError, IndexError) as e:
        return e
