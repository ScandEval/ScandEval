"""Functions related to text generation of models."""

import json
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollator, GenerationConfig, StoppingCriteria
from transformers.modeling_utils import ModelOutput

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .openai_models import OpenAIModel
from .protocols import GenerativeModel, Tokenizer
from .utils import clear_memory

logger = logging.getLogger(__package__)


def generate(
    itr: tqdm,
    train: Dataset,
    val: Dataset,
    tests: list[Dataset],
    prepared_train: Dataset,
    prepared_val: Dataset,
    prepared_tests: list[Dataset],
    model: GenerativeModel,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    data_collator: DataCollator,
    compute_metrics: Callable,
    extract_labels_fn: Callable[..., list[str]],
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
) -> dict[str, list[dict[str, float]]]:
    """Evaluate a model on a dataset through generation.

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
        num_iter:
            The number of iterations to run.
        rng:
            The random number generator.
        model:
            The model to evaluate.
        model_config:
            The configuration of the model.
        tokenizer:
            The tokenizer to use for the model. If `None` then the model's
            tokenizer will be used.
        data_collator:
            The data collator to use for the model.
        compute_metrics:
            The function to use to compute the metrics.
        extract_labels_fn:
            The function to use to extract the labels from the model output.
        benchmark_config:
            The configuration of the benchmark.
        dataset_config:
            The configuration of the dataset.

    Returns:
        A dictionary containing the scores, with keys "test" and maybe "train", with
        values being lists of dicts containing the scores for each metric for each
        iteration.
    """
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)

    # Create model output cache
    model_cache_dir = Path(model_config.model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_cache_dir / "model_outputs.json"
    if not cache_path.exists():
        with cache_path.open("w") as f:
            json.dump(dict(), f)

    for idx in itr:
        prepared_test = prepared_tests[idx]
        assert isinstance(prepared_test, Dataset)

        while True:
            try:
                test_scores = generate_single_iteration(
                    prepared_dataset=prepared_test,
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    extract_labels_fn=extract_labels_fn,
                    benchmark_config=benchmark_config,
                    dataset_config=dataset_config,
                    model_cache_dir=Path(model_config.model_cache_dir),
                )
                break
            except Exception as e:
                oom_error = [
                    "CUDA out of memory",
                    "CUDA error",
                    "MPS backend out of memory",
                    "Too many parallel completions requested.",  # OpenAI specific
                ]
                if all(error not in str(e) for error in oom_error):
                    raise InvalidBenchmark(str(e))
                clear_memory()
                benchmark_config.batch_size //= 2
                if benchmark_config.batch_size < 1:
                    raise InvalidBenchmark(
                        "GPU out of memory, even with a batch size of 1!"
                    )

        logger.debug(f"Test scores for iteration {idx}: {test_scores}")
        scores["test"].append(test_scores)
        clear_memory()

        if benchmark_config.evaluate_train:
            train_scores = generate_single_iteration(
                prepared_dataset=prepared_train,
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                extract_labels_fn=extract_labels_fn,
                benchmark_config=benchmark_config,
                dataset_config=dataset_config,
                model_cache_dir=Path(model_config.model_cache_dir),
            )
            logger.debug(f"Train scores for iteration {idx}: {train_scores}")
            scores["train"].append(train_scores)
            clear_memory()

    return scores


def generate_single_iteration(
    prepared_dataset: Dataset,
    model: GenerativeModel,
    tokenizer: Tokenizer,
    data_collator: DataCollator,
    compute_metrics: Callable,
    extract_labels_fn: Callable[..., list[Any]],
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
    model_cache_dir: Path,
) -> dict[str, float]:
    """Evaluate a model on a dataset in a single iteration through generation.

    Args:
        prepared_dataset:
            The dataset to evaluate on.
        model:
            The model to evaluate.
        tokenizer:
            The tokenizer to use for the model.
        data_collator:
            The data collator to use for the model.
        compute_metrics:
            The function to use to compute the metrics.
        extract_labels_fn:
            The function to use to extract the labels from the dataset.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
        model_cache_dir:
            The directory to store the model output cache in.

    Returns:
        A list of dictionaries containing the scores for each metric.
    """
    # If we are testing then we save the model outputs to a different cache and ensure
    # that that cache is deleted before the next test, to ensure that the tests are
    # independent of each other
    if not hasattr(sys, "_called_from_test"):
        cache_name = "model_outputs.json"
    else:
        cache_name = "model_outputs_test.json"
        (model_cache_dir / cache_name).unlink(missing_ok=True)

    cache = ModelCache(model_cache_dir=model_cache_dir, cache_name=cache_name)

    # Split up the prepared dataset into a cached and non-cached part
    cached_dataset, non_cached_dataset = split_dataset_into_cached_and_non_cached(
        dataset=prepared_dataset, cache=cache
    )

    all_preds: list[str | list[str]] = list()

    if len(non_cached_dataset) > 0:
        # Tokens used in generation to know when generation is finished
        stopping_criteria = get_generation_stopping_criteria(
            tokenizer=tokenizer, model=model
        )

        generation_config = GenerationConfig(
            max_new_tokens=dataset_config.max_generated_tokens,
            do_sample=False,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            return_dict_in_generate=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Sort the non_cached dataset by the length of the text, to minimise the amount
        # of padding that needs to be added, speeding up generation
        non_cached_dataset = non_cached_dataset.add_column(
            name="length", column=[len(x) for x in non_cached_dataset["text"]]
        )
        non_cached_dataset = non_cached_dataset.sort("length", reverse=False)

        # Enable batching by building a dataloader. The dataloader cannot deal with
        # text columns, so we create a copy of the dataset without these
        torch_dataset = non_cached_dataset.with_format("torch").remove_columns(
            [
                column
                for column in non_cached_dataset.column_names
                if column != "input_ids"
            ]
        )

        batch_size = (
            1 if isinstance(model, OpenAIModel) else benchmark_config.batch_size
        )
        dataloader = DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=data_collator,
        )

        # Generate the completions for the non-cached examples
        for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
            # Generate the completions of the documents in the batch
            with warnings.catch_warnings(), torch.inference_mode():
                warnings.simplefilter("ignore", category=UserWarning)
                inputs = batch["input_ids"].to(model.device)
                model_output: ModelOutput = model.generate(
                    inputs=inputs, generation_config=generation_config
                )

            # Extract the scores from the model output, to be cached. We only store the
            # indices of the top 100 scores, to save space
            if "scores" in model_output:
                scores = torch.stack(model_output.scores, dim=1)
                top_scores = torch.topk(scores, k=100)

            # Store the generated sequences in the cache, one by one
            for sample_idx, sample in enumerate(inputs):
                decoded_inputs = tokenizer.decode(
                    token_ids=sample, skip_special_tokens=True
                )
                generated_ids = model_output.sequences[sample_idx]

                # Hugging Face models include the input in the generated sequence, so we
                # need to remove it in that case
                if torch.equal(generated_ids[: sample.shape[0]], sample):
                    generated_ids = generated_ids[sample.shape[0] :]

                generated_ids = generated_ids.tolist()

                # Store the generated sequence in the cache
                cached_model_output = GenerativeModelOutput(
                    completion=tokenizer.decode(
                        token_ids=generated_ids, skip_special_tokens=True
                    ),
                )
                if "scores" in model_output:
                    cached_model_output.top_score_indices = top_scores.indices[
                        sample_idx
                    ].tolist()
                    cached_model_output.top_score_values = top_scores.values[
                        sample_idx
                    ].tolist()
                    cached_model_output.vocab_size = int(scores.shape[-1])
                cache[decoded_inputs] = cached_model_output

            # Extract the labels from the model output and store them for metric
            # computation later
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            input_batch = non_cached_dataset[batch_start:batch_end]
            extracted_labels: list = extract_labels_fn(
                input_batch=input_batch, model_output=model_output, tokenizer=tokenizer
            )
            all_preds.extend(extracted_labels)

        cache.save()

    # Fetch the cached predictions for the cached examples
    if len(cached_dataset) > 0:
        model_output = load_cached_model_outputs(
            cached_dataset=cached_dataset,
            cache=cache,
            tokenizer=tokenizer,
        )
        extracted_labels = extract_labels_fn(
            input_batch=cached_dataset,
            model_output=model_output,
            tokenizer=tokenizer,
        )
        all_preds.extend(extracted_labels)

    if "label" in non_cached_dataset.column_names:
        ground_truth = [
            label.lower() if isinstance(label, str) else label
            for label in non_cached_dataset["label"] + cached_dataset["label"]
        ]
    elif "labels" in non_cached_dataset.column_names:
        ground_truth = [
            [label.lower() if isinstance(label, str) else label for label in label_list]
            for label_list in non_cached_dataset["labels"] + cached_dataset["labels"]
        ]
    elif "target_text" in non_cached_dataset.column_names:
        ground_truth = non_cached_dataset["target_text"] + cached_dataset["target_text"]
    else:
        raise ValueError(
            "The dataset must have either a 'label', 'labels', or 'target_text' column"
        )

    itr_scores: dict[str, float] = compute_metrics(
        model_outputs_and_labels=(all_preds, ground_truth),
        id2label=dataset_config.id2label,
    )

    return itr_scores


@dataclass
class GenerativeModelOutput:
    completion: str
    top_score_indices: list[list[int]] | None = None
    top_score_values: list[list[float]] | None = None
    vocab_size: int | None = None


class ModelCache:
    """A cache for model outputs.

    Args:
        model_cache_dir:
            The directory to store the cache in.
        cache_name:
            The name of the cache file.

    Attributes:
        model_cache_dir:
            The directory to store the cache in.
        cache_path:
            The path to the cache file.
        cache:
            The model output cache.
    """

    def __init__(
        self,
        model_cache_dir: Path,
        cache_name: str,
    ):
        self.model_cache_dir = model_cache_dir
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.model_cache_dir / cache_name
        self.cache: dict[str, GenerativeModelOutput] = self.load()

    def load(self) -> dict[str, GenerativeModelOutput]:
        """Load and return the model output cache."""
        if not self.cache_path.exists():
            with self.cache_path.open("w") as f:
                json.dump(dict(), f)

        with self.cache_path.open() as f:
            json_cache = json.load(f)

        cache: dict[str, GenerativeModelOutput] = dict()
        for key in json_cache:
            cache[key] = GenerativeModelOutput(**json_cache[key])
        return cache

    def save(self) -> None:
        """Save the model output cache to disk."""
        dumpable_cache: dict[str, dict] = defaultdict(dict)
        for key, value in self.cache.items():
            dumpable_cache[key] = asdict(value)

        with self.cache_path.open("w") as f:
            json.dump(dumpable_cache, f, indent=4)

    def __getitem__(self, key: str) -> GenerativeModelOutput:
        """Get an item from the cache.

        Args:
            key:
                The key to use to index the cache.

        Returns:
            The model output.
        """
        return self.cache[key]

    def __setitem__(self, key: str, value: GenerativeModelOutput) -> None:
        """Set an item in the cache.

        Args:
            key:
                The key to use to index the cache.
            value:
                The value to set in the cache.
        """
        self.cache[key] = value

    def cached_texts(self) -> list[str]:
        """Return the text inputs indexed in the cache."""
        return [key for key in self.cache.keys()]


def split_dataset_into_cached_and_non_cached(
    dataset: Dataset, cache: ModelCache
) -> tuple[Dataset, Dataset]:
    """Split a dataset into a cached and non-cached part.

    Args:
        dataset:
            The dataset to split.
        cache:
            The model output cache.

    Returns:
        The cached and non-cached parts of the dataset.
    """
    # Get the sample indices of the non-cached examples, which are unique with respect
    # to the "text" column
    unique_non_cached_ids: set[int] = set()
    for example_idx, example in enumerate(dataset):
        cached_texts = cache.cached_texts()
        cached_texts += dataset.select(unique_non_cached_ids)["text"]
        if example["text"] not in cached_texts:
            unique_non_cached_ids.add(example_idx)

    # The cached examples are the ones that are not in the non-cached examples. This
    # means that if the dataset has duplicates, only a single copy of the duplicate
    # will be put in the non-cached part, and the rest in the cached part.
    cached_ids = set(range(len(dataset))) - unique_non_cached_ids

    cached = dataset.select(cached_ids)
    non_cached = dataset.select(unique_non_cached_ids)
    return cached, non_cached


def load_cached_model_outputs(
    cached_dataset: Dataset,
    cache: ModelCache,
    tokenizer: Tokenizer,
) -> ModelOutput:
    """Load the cached model outputs.

    Args:
        cached_dataset:
            The dataset containing the cached examples.
        cache:
            The model output cache.
        tokenizer:
            The tokenizer used to generate the tokens.

    Returns:
        The model output containing the cached sequences.
    """
    # Encode and decode the samples in the cached dataset, to ensure that the cached
    # sequences are tokenized in the same way as the model outputs
    cached_dataset = cached_dataset.map(
        lambda example: dict(
            text=tokenizer.decode(
                tokenizer(
                    text=example["text"],
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.squeeze(dim=0),
                skip_special_tokens=True,
            )
        ),
    )

    # Load the raw model outputs from the cache
    cached_model_outputs: list[GenerativeModelOutput] = [
        cache[example["text"]] for example in cached_dataset
    ]

    # Tokenize the cached sequences
    tokenized_cached_sequences: list[torch.Tensor] = [
        tokenizer(
            text=cached_model_output.completion,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(dim=0)
        for cached_model_output in cached_model_outputs
    ]

    # Pad the cached completions to the same length
    cached_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences=tokenized_cached_sequences,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    # If we do not have any cached scores, then wrap the padded cached sequences in a
    # ModelOutput and return it
    if (
        cached_model_outputs[0].top_score_indices is None
        or cached_model_outputs[0].top_score_values is None
        or cached_model_outputs[0].vocab_size is None
    ):
        return ModelOutput(sequences=cached_sequences)

    # Otherwise, we format the cached scores into a tensor of shape [batch_size,
    # num_sequences, vocab_size], wrap it in a ModelOutput with the padded cached
    # sequences, and return it
    cached_scores = torch.zeros(
        len(cached_model_outputs),
        len(cached_model_outputs[0].top_score_indices),
        cached_model_outputs[0].vocab_size,
    )
    top_score_indices = torch.tensor(
        [
            cached_model_output.top_score_indices
            for cached_model_output in cached_model_outputs
        ]
    )
    top_score_values = torch.tensor(
        [
            cached_model_output.top_score_values
            for cached_model_output in cached_model_outputs
        ]
    )
    for batch_idx in range(cached_scores.shape[0]):
        for sequence_idx in range(cached_scores.shape[1]):
            top_indices = top_score_indices[batch_idx, sequence_idx]
            top_values = top_score_values[batch_idx, sequence_idx]
            cached_scores[batch_idx, sequence_idx, top_indices] = top_values
    return ModelOutput(sequences=cached_sequences, scores=cached_scores)


def extract_raw_predictions(
    generated_sequences: torch.Tensor,
    tokenizer: Tokenizer,
    dataset_config: DatasetConfig,
) -> list[str]:
    """Get the raw predictions from the generated sequences.

    Args:
        generated_sequences:
            The generated sequences from the model. The outer-most list is the
            batch dimension, the inner-most list is the sequence dimension,
            consisting of token IDs.
        tokenizer:
            The tokenizer used to generate the tokens.
        dataset_config:
            The dataset config.

    Returns:
        The candidate labels with the smallest edit distance to the predicted labels.
    """
    completion_ids_lists = [
        [int(token_id) for token_id in completion_ids]
        for completion_ids in generated_sequences
    ]

    # For some models the generated tokens also includes the input tokens, so we need
    # to deal with both cases when extracting the predicted labels
    prompt_prefix_exists = dataset_config.prompt_prefix != ""
    pred_idx = dataset_config.num_few_shot_examples + int(prompt_prefix_exists)
    raw_predictions: list[str] = list()
    for completion_ids_list in completion_ids_lists:
        decoded = tokenizer.decode(completion_ids_list, skip_special_tokens=True)
        few_shots = decoded.split("\n\n")
        answer_exists = len(few_shots) > pred_idx
        answer = few_shots[pred_idx] if answer_exists else few_shots[-1]
        answer = answer.split("\n")[-1]
        answer = answer.strip()
        raw_predictions.append(answer)
    return raw_predictions


def get_generation_stopping_criteria(
    tokenizer: Tokenizer,
    model: GenerativeModel,
) -> list[StoppingCriteria]:
    """Get the stopping criteria for generation.

    Args:
        tokenizer:
            The tokenizer used to tokenize the stop words.
        model:
            The generative model, which we use to ensure the tensors are on the
            same device, and also determine whether stop words are needed, based on
            the model type.

    Returns:
        The stopping criteria for generation.
    """
    if isinstance(model, OpenAIModel):
        return list()

    double_newline_ids: list[int] = tokenizer(
        text=["\n\n"], add_special_tokens=False
    ).input_ids[0]
    single_newline_ids: list[int] = tokenizer(
        text=["\n"], add_special_tokens=False
    ).input_ids[0]
    bos_token_ids: list[int] = tokenizer(
        text=[tokenizer.bos_token], add_special_tokens=False
    ).input_ids[0]
    eos_token_ids: list[int] = tokenizer(
        text=[tokenizer.eos_token], add_special_tokens=False
    ).input_ids[0]

    def remove_empty_tokens(token_id_list: list[int]) -> list[int]:
        return [
            token_id for token_id in token_id_list if tokenizer.decode([token_id]) != ""
        ]

    double_newline_ids = remove_empty_tokens(double_newline_ids)
    single_newline_ids = remove_empty_tokens(single_newline_ids)
    bos_token_ids = remove_empty_tokens(bos_token_ids)
    eos_token_ids = remove_empty_tokens(eos_token_ids)

    stop_word_id_lists = [
        double_newline_ids,
        single_newline_ids + single_newline_ids,
        bos_token_ids,
        eos_token_ids,
    ]

    return [StopWordCriteria(stop_word_id_lists=stop_word_id_lists)]


class StopWordCriteria(StoppingCriteria):
    def __init__(self, stop_word_id_lists: list[list[int]]):
        super().__init__()
        self.stop_word_id_lists = stop_word_id_lists

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for stop_word_id_list in self.stop_word_id_lists:
            if stop_word_id_list == input_ids[0][-len(stop_word_id_list) :].tolist():
                return True
        return False
