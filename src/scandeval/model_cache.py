"""ModelCache class for caching model outputs."""

import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers.modeling_utils import ModelOutput

from .protocols import Tokenizer

logger = logging.getLogger(__package__)


@dataclass
class GenerativeModelOutput:
    """The output of a generative model."""

    completion: str
    top_score_indices: list[list[int]] | None = None
    top_score_values: list[list[float]] | None = None
    vocab_size: int | None = None


class ModelCache:
    """A cache for model outputs.

    Attributes:
        model_cache_dir:
            The directory to store the cache in.
        cache_path:
            The path to the cache file.
        cache:
            The model output cache.
        max_generated_tokens:
            The maximum number of tokens to generate for each example.
    """

    def __init__(
        self, model_cache_dir: Path, cache_name: str, max_generated_tokens: int
    ):
        """Initialize the model output cache.

        Args:
            model_cache_dir:
                The directory to store the cache in.
            cache_name:
                The name of the cache file.
            max_generated_tokens:
                The maximum number of tokens to generate for each example.
        """
        self.model_cache_dir = model_cache_dir
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.model_cache_dir / cache_name
        self.max_generated_tokens = max_generated_tokens

    def load(self) -> None:
        """Load the model output cache."""
        if not self.cache_path.exists():
            with self.cache_path.open("w") as f:
                json.dump(dict(), f)

        with self.cache_path.open() as f:
            json_cache = json.load(f)

        cache: dict[str, GenerativeModelOutput] = dict()
        for key in json_cache:
            cache[key] = GenerativeModelOutput(**json_cache[key])

        self.cache = cache

    def save(self) -> None:
        """Save the model output cache to disk."""
        dumpable_cache: dict[str, dict] = defaultdict(dict)
        for key, value in self.cache.items():
            dumpable_cache[key] = asdict(value)

        with self.cache_path.open("w") as f:
            json.dump(dumpable_cache, f)

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

    def remove(self) -> None:
        """Remove the cache from memory and delete it from disk."""
        self.cache_path.unlink()
        del self.cache

    def cached_texts(self) -> list[str]:
        """Return the text inputs indexed in the cache."""
        return [key for key in self.cache.keys()]

    def add_to_cache(
        self, model_input: torch.Tensor, model_output: ModelOutput, tokenizer: Tokenizer
    ) -> None:
        """Add the model input/output to the cache.

        Args:
            model_input:
                The model input.
            model_output:
                The model output.
            tokenizer:
                The tokenizer used to generate the tokens.
        """
        model_input = model_input.detach().cpu()

        # Extract the scores from the model output, to be cached. We only store the
        # indices of the top scores, to save space. Further, we only store the scores
        # if the generated sequence is shorter than the maximum length
        store_scores = "scores" in model_output and self.max_generated_tokens < 8
        if store_scores:
            scores = torch.stack(
                tensors=[
                    score_tensor.detach().cpu() for score_tensor in model_output.scores
                ],
                dim=1,
            )
            top_scores = torch.topk(scores, k=10)

        # Store the generated sequences in the cache, one by one
        # TODO: This is a bit slow, should be optimized
        with tqdm(
            iterable=model_input,
            desc="Caching model outputs",
            leave=False,
            disable=hasattr(sys, "_called_from_test"),
        ) as pbar:
            for sample_idx, sample in enumerate(pbar):
                decoded_inputs = tokenizer.decode(
                    token_ids=sample, skip_special_tokens=True
                )
                generated_ids = model_output.sequences[sample_idx].tolist()

                # Set up the model output in a GenerativeModelOutput object
                cached_model_output = GenerativeModelOutput(
                    completion=tokenizer.decode(
                        token_ids=generated_ids, skip_special_tokens=True
                    )
                )
                if store_scores:
                    cached_model_output.top_score_indices = top_scores.indices[
                        sample_idx
                    ].tolist()
                    cached_model_output.top_score_values = top_scores.values[
                        sample_idx
                    ].tolist()
                    cached_model_output.vocab_size = int(scores.shape[-1])

                # Store the generated sequence in the cache
                self[decoded_inputs] = cached_model_output


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
    # to the "text" column.
    dataset_texts = pd.Series(dataset["text"])
    dataset_texts.drop_duplicates(inplace=True)
    unique_non_cached_ids = set(
        dataset_texts[~dataset_texts.isin(cache.cached_texts())].index.tolist()
    )

    # The cached examples are the ones that are not in the non-cached examples. This
    # means that if the dataset has duplicates, only a single copy of the duplicate
    # will be put in the non-cached part, and the rest in the cached part.
    cached_ids = set(range(len(dataset))) - unique_non_cached_ids

    cached = dataset.select(cached_ids)
    non_cached = dataset.select(unique_non_cached_ids)
    return cached, non_cached


def load_cached_model_outputs(
    cached_dataset: Dataset, cache: ModelCache, tokenizer: Tokenizer
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
    # Load the raw model outputs from the cache
    cached_model_outputs: list[GenerativeModelOutput] = [
        cache[prompt] for prompt in cached_dataset["text"]
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
        max(
            len(cached_model_output.top_score_indices)
            for cached_model_output in cached_model_outputs
            if cached_model_output.top_score_indices is not None
        ),
        cached_model_outputs[0].vocab_size,
    )
    top_score_indices = torch.nn.utils.rnn.pad_sequence(
        sequences=[
            torch.tensor(cached_model_output.top_score_indices)
            for cached_model_output in cached_model_outputs
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    top_score_values = torch.nn.utils.rnn.pad_sequence(
        sequences=[
            torch.tensor(cached_model_output.top_score_values)
            for cached_model_output in cached_model_outputs
        ],
        batch_first=True,
        padding_value=0.0,
    )
    for batch_idx in range(cached_scores.shape[0]):
        for sequence_idx in range(cached_scores.shape[1]):
            top_indices = top_score_indices[batch_idx, sequence_idx]
            top_values = top_score_values[batch_idx, sequence_idx]
            cached_scores[batch_idx, sequence_idx, top_indices] = top_values
    return ModelOutput(sequences=cached_sequences, scores=cached_scores)
