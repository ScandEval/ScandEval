"""Functions related to text generation of models."""

import itertools as it
import logging
import warnings
from collections import defaultdict
from typing import Callable

import Levenshtein
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.cuda import OutOfMemoryError
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollator, GenerationConfig, StoppingCriteria
from transformers.modeling_utils import ModelOutput

from .config import BenchmarkConfig, DatasetConfig
from .exceptions import InvalidBenchmark
from .model_setups import GenerativeModel, Tokenizer
from .openai_models import OpenAIModel
from .utils import clear_memory

logger = logging.getLogger(__name__)


def generate(
    itr: tqdm,
    train: Dataset,
    val: Dataset,
    tests: list[Dataset],
    prepared_train: Dataset,
    prepared_val: Dataset,
    prepared_tests: list[Dataset],
    model: GenerativeModel,
    tokenizer: Tokenizer,
    data_collator: DataCollator,
    compute_metrics: Callable,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
) -> dict[str, list[dict[str, float]]]:
    """Evaluate a model on a dataset through generation.

    Args:
        itr (tqdm.tqdm):
            The progress bar iterator.
        train (Dataset):
            The training dataset.
        val (Dataset):
            The validation dataset.
        tests (list[Dataset]):
            The bootstrapped test datasets.
        prepared_train (Dataset):
            The prepared training dataset.
        prepared_val (Dataset):
            The prepared validation dataset.
        prepared_tests (list[Dataset]):
            The prepared bootstrapped test datasets.
        num_iter (int):
            The number of iterations to run.
        rng (np.random.Generator):
            The random number generator.
        model (GenerativeModel):
            The model to evaluate.
        tokenizer (Tokenizer):
            The tokenizer to use for the model. If `None` then the model's
            tokenizer will be used.
        data_collator (DataCollator):
            The data collator to use for the model.
        compute_metrics (Callable):
            The function to use to compute the metrics.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.
        dataset_config (DatasetConfig):
            The configuration of the dataset.

    Returns:
        dict[str, list[dict[str, float]]]:
            A dictionary containing the scores, with keys "test" and maybe "train",
            with values being lists of dicts containing the scores for each metric
            for each iteration.
    """
    scores: dict[str, list[dict[str, float]]] = defaultdict(list)

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
                    benchmark_config=benchmark_config,
                    dataset_config=dataset_config,
                )
                break
            except OutOfMemoryError:
                clear_memory()
                benchmark_config.batch_size //= 2

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
                benchmark_config=benchmark_config,
                dataset_config=dataset_config,
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
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> dict[str, float]:
    """Evaluate a model on a dataset in a single iteration through generation.

    Args:
        prepared_dataset (Dataset):
            The dataset to evaluate on.
        model (GenerativeModel):
            The model to evaluate.
        tokenizer (Tokenizer):
            The tokenizer to use for the model.
        data_collator (DataCollator):
            The data collator to use for the model.
        compute_metrics (Callable):
            The function to use to compute the metrics.
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Returns:
        list[dict[str, float]]:
            A list of dictionaries containing the scores for each metric.
    """
    # Tokens used in generation to know when generation is finished
    stopping_criteria = get_generation_stopping_criteria(
        tokenizer=tokenizer, model=model
    )

    generation_config = GenerationConfig(
        max_new_tokens=dataset_config.max_generated_tokens,
        temperature=0.0,
        do_sample=False,
        stopping_criteria=stopping_criteria,
        output_scores=True,
        return_dict_in_generate=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Find the largest batch size that fits in memory
    max_seq_len_in_dataset = max(map(len, prepared_dataset["input_ids"]))
    max_input_length_plus_buffer = min(
        int(max_seq_len_in_dataset * 1.1), tokenizer.model_max_length
    )
    batch_sizes: list[int] = [
        benchmark_config.batch_size // (2**n)
        for n in range(1 + np.log2(benchmark_config.batch_size).astype(int))
    ]
    batch_size = batch_sizes[0]
    for batch_size in batch_sizes:
        dummy_inputs = torch.full(
            size=(batch_size, max_input_length_plus_buffer),
            fill_value=tokenizer.pad_token_id,
            device=model.device,
            dtype=torch.long,
        )
        try:
            with torch.no_grad():
                model.generate(dummy_inputs, generation_config=generation_config)
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
            continue
    else:
        raise InvalidBenchmark("GPU out of memory, even with a batch size of 1!")
    benchmark_config.batch_size = batch_size

    # Sort the dataset by the length of the text, to minimise the amount of padding
    # that needs to be added, speeding up generation
    text_column = "text" if "text" in prepared_dataset.column_names else "doc"
    prepared_dataset = prepared_dataset.add_column(
        name="length", column=[len(x) for x in prepared_dataset[text_column]]
    )
    prepared_dataset = prepared_dataset.sort("length", reverse=True)

    # Enable batching by building a dataloader. The dataloader cannot deal with
    # text columns, so we create a copy of the dataset without these
    torch_dataset = prepared_dataset.with_format("torch").remove_columns(
        [column for column in prepared_dataset.column_names if column != "input_ids"]
    )

    all_preds: list[str] = list()

    dataloader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=data_collator,
    )

    for batch in tqdm(dataloader, leave=False):
        # Generate the completions of the documents in the batch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            inputs = batch["input_ids"].to(model.device)
            with torch.no_grad():
                model_output: ModelOutput = model.generate(
                    inputs=inputs, generation_config=generation_config
                )

        # Extract the predicted labels from the model output
        if dataset_config.task.supertask == "sequence-classification":
            if "scores" in model_output:
                predicted_labels = get_closest_logprobs_labels(
                    generation_logprobs=model_output["scores"],
                    tokenizer=tokenizer,
                    dataset_config=dataset_config,
                )
            else:
                predicted_labels = get_closest_word_edit_labels(
                    generated_sequences=model_output["sequences"],
                    tokenizer=tokenizer,
                    dataset_config=dataset_config,
                )

        all_preds.extend(predicted_labels)

    true_labels = [
        dataset_config.prompt_label_mapping[lbl] for lbl in prepared_dataset["label"]
    ]

    itr_scores = compute_metrics(
        model_outputs_and_labels=(all_preds, true_labels),
        id2label=dataset_config.id2label,
    )

    return itr_scores


def get_closest_logprobs_labels(
    generation_logprobs: tuple[torch.Tensor],
    tokenizer: Tokenizer,
    dataset_config: DatasetConfig,
) -> list[str]:
    """Get the labels with the highest predicted logprob value.

    In case a candidate label is split into multiple tokens, we only use the first
    token to compute the logprob value. E.g., if the candidate label "positive" is
    tokenised as ["pos", "itive"], we only use the logprob value of "pos" to
    represent the logprob value of the entire label.

    Args:
        generation_logprobs (tuple[torch.Tensor]):
            The logprobs of the generated tokens.
        tokenizer (Tokenizer):
            The tokenizer used to generate the tokens.
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Returns:
        list[str]:
            The predicted labels.
    """
    candidate_labels = [
        dataset_config.prompt_label_mapping[lbl] for lbl in dataset_config.id2label
    ]

    # Shape: [batch_size, num_generated_tokens, vocab_size]
    all_logprobs = torch.stack(generation_logprobs, dim=1)

    # Shape: [batch_size, num_candidate_labels]
    pred_logprobs = torch.empty(
        all_logprobs.shape[0], len(candidate_labels), device=all_logprobs.device
    )

    for idx, candidate_label in enumerate(candidate_labels):
        # We only use the first token to represent the logprob value of the entire
        # label.
        candidate_label_ids: list[list[int]] = tokenizer(
            [candidate_label.lower()], add_special_tokens=False
        )["input_ids"]
        candidate_label_id: int = candidate_label_ids[0][0]
        pred_logprobs[:, idx] = all_logprobs[:, 0, candidate_label_id]

    # Shape: [batch_size,]
    predicted_label_ids = pred_logprobs.argmax(dim=1)

    return [candidate_labels[idx] for idx in predicted_label_ids]


def get_closest_word_edit_labels(
    generated_sequences: list[list[int]],
    tokenizer: Tokenizer,
    dataset_config: DatasetConfig,
) -> list[str]:
    """Get the labels with the smallest edit distance to the predicted labels.

    Args:
        generated_sequences (list of list of int):
            The generated sequences from the model. The outer-most list is the
            batch dimension, the inner-most list is the sequence dimension,
            consisting of token IDs.
        tokenizer (Tokenizer):
            The tokenizer used to generate the tokens.
        dataset_config (DatasetConfig):
            The configuration of the dataset.

    Returns:
        list of str:
            The candidate labels with the smallest edit distance to the predicted
            labels.
    """
    raw_predictions = extract_raw_predictions(
        generated_sequences=generated_sequences,
        tokenizer=tokenizer,
        dataset_config=dataset_config,
    )

    candidate_labels = dataset_config.id2label
    new_predicted_labels: list[str] = list()
    for predicted_label in raw_predictions:
        edit_distances = [
            Levenshtein.distance(s1=predicted_label.lower(), s2=candidate_label.lower())
            for candidate_label in candidate_labels
        ]
        closest_label = candidate_labels[np.argmin(edit_distances).item()]
        new_predicted_labels.append(closest_label)
    return new_predicted_labels


def extract_raw_predictions(
    generated_sequences: list[list[int]],
    tokenizer: Tokenizer,
    dataset_config: DatasetConfig,
) -> list[str]:
    """Get the labels with the smallest edit distance to the predicted labels.

    Args:
        generated_sequences (list of list of int):
            The generated sequences from the model. The outer-most list is the
            batch dimension, the inner-most list is the sequence dimension,
            consisting of token IDs.
        tokenizer (Tokenizer):
            The tokenizer used to generate the tokens.
        dataset_config (DatasetConfig):
            The dataset config.

    Returns:
        list of str:
            The candidate labels with the smallest edit distance to the predicted
            labels.
    """

    completion_ids_lists = [
        [
            token_id
            for token_id in completion_ids
            if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id]
        ]
        for completion_ids in generated_sequences
    ]

    # For some models the generated tokens also includes the input tokens, so
    # we need to deal with both cases when extracting the predicted labels
    try:
        pred_idx = dataset_config.num_few_shot_examples + 1
        return [
            tokenizer.decode(completion_ids_list)
            .split("\n\n")[pred_idx]
            .split("\n")[-1]
            .split(":")[-1]
            .strip()
            for completion_ids_list in completion_ids_lists
        ]
    except IndexError:
        return [
            tokenizer.decode(completion_ids_list).strip()
            for completion_ids_list in completion_ids_lists
        ]


def get_generation_stopping_criteria(
    tokenizer: Tokenizer,
    model: GenerativeModel,
) -> list[StoppingCriteria]:
    """Get the stopping criteria for generation.

    Args:
        tokenizer (Tokenizer):
            The tokenizer used to tokenize the stop words.
        model (GenerativeModel):
            The generative model, which we use to ensure the tensors are on the
            same device, and also determine whether stop words are needed, based on
            the model type.

    Returns:
        list[torch.Tensor]:
            A list of tensors containing the stop words to use for generation.
    """
    if isinstance(model, OpenAIModel):
        return list()

    stop_word_ids: list[torch.Tensor] = list()

    double_newline_ids: torch.Tensor = (
        tokenizer(
            text=["\n\n"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .to(model.device)
    )
    single_newline_ids: torch.Tensor = (
        tokenizer(
            text=["\n"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .to(model.device)
    )
    bos_token_ids: torch.Tensor = (
        tokenizer(
            text=[tokenizer.bos_token],
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .to(model.device)
    )
    eos_token_ids: torch.Tensor = (
        tokenizer(
            text=[tokenizer.eos_token],
            add_special_tokens=False,
            return_tensors="pt",
        )
        .input_ids[0]
        .to(model.device)
    )

    double_newline_ids = double_newline_ids[
        [tokenizer.decode(tok) != "" for tok in double_newline_ids]
    ]
    single_newline_ids = single_newline_ids[
        [tokenizer.decode(tok) != "" for tok in single_newline_ids]
    ]
    bos_token_ids = bos_token_ids[
        [tokenizer.decode(tok) != "" for tok in bos_token_ids]
    ]
    eos_token_ids = eos_token_ids[
        [tokenizer.decode(tok) != "" for tok in eos_token_ids]
    ]

    two_single_newline_ids = torch.cat([single_newline_ids, single_newline_ids], dim=0)

    stop_word_ids = [
        double_newline_ids,
        two_single_newline_ids,
        bos_token_ids,
        eos_token_ids,
    ]

    return [StopWordCriteria(stop_word_ids=stop_word_ids)]


def extract_few_shot_examples(
    shuffled_train: Dataset,
    dataset_config: DatasetConfig,
) -> list[dict]:
    supertask = dataset_config.task.supertask
    num_few_shots = dataset_config.num_few_shot_examples
    if supertask == "sequence-classification":
        labels = it.cycle(dataset_config.task.labels)
        few_shot_examples: list[dict] = list()
        while len(few_shot_examples) < num_few_shots:
            label = next(labels)
            examples = shuffled_train.filter(
                lambda x: x["label"].lower() == label.lower()
            ).select(range(1))
            few_shot_examples.append(examples[0])
    else:
        examples_df = shuffled_train.select(range(num_few_shots)).to_pandas()
        assert isinstance(examples_df, pd.DataFrame)
        few_shot_examples = examples_df.to_dict("records")
    return few_shot_examples


class StopWordCriteria(StoppingCriteria):
    def __init__(self, stop_word_ids: list[torch.Tensor]):
        super().__init__()
        self.stop_word_ids = stop_word_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for stop_word_tensor in self.stop_word_ids:
            if torch.all(
                (stop_word_tensor == input_ids[0][-len(stop_word_tensor) :])
            ).item():
                return True
        return False
