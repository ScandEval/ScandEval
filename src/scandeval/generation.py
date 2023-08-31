"""Functions related to text generation of models."""

import logging
import warnings
from collections import defaultdict
from typing import Any, Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollator, GenerationConfig, StoppingCriteria
from transformers.modeling_utils import ModelOutput

from .config import BenchmarkConfig, DatasetConfig
from .exceptions import InvalidBenchmark
from .model_setups import GenerativeModel, Tokenizer
from .openai_models import OpenAIModel
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

    Returns:
        A list of dictionaries containing the scores for each metric.
    """
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

    # Sort the dataset by the length of the text, to minimise the amount of padding
    # that needs to be added, speeding up generation
    prepared_dataset = prepared_dataset.add_column(
        name="length", column=[len(x) for x in prepared_dataset["text"]]
    )
    prepared_dataset = prepared_dataset.sort("length", reverse=False)

    # Enable batching by building a dataloader. The dataloader cannot deal with
    # text columns, so we create a copy of the dataset without these
    torch_dataset = prepared_dataset.with_format("torch").remove_columns(
        [column for column in prepared_dataset.column_names if column != "input_ids"]
    )
    all_preds: list[str | list[str]] = list()

    batch_size = 1 if isinstance(model, OpenAIModel) else benchmark_config.batch_size
    dataloader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=data_collator,
    )

    for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
        # Generate the completions of the documents in the batch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            inputs = batch["input_ids"].to(model.device)
            with torch.inference_mode():
                model_output: ModelOutput = model.generate(
                    inputs=inputs, generation_config=generation_config
                )

        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        input_batch = prepared_dataset[batch_start:batch_end]
        extracted_labels: list = extract_labels_fn(
            input_batch=input_batch, model_output=model_output, tokenizer=tokenizer
        )
        all_preds.extend(extracted_labels)

    if "label" in prepared_dataset.column_names:
        true_labels = [
            label.lower() if isinstance(label, str) else label
            for label in prepared_dataset["label"]
        ]
    else:
        true_labels = [
            [label.lower() if isinstance(label, str) else label for label in label_list]
            for label_list in prepared_dataset["labels"]
        ]

    itr_scores: dict[str, float] = compute_metrics(
        model_outputs_and_labels=(all_preds, true_labels),
        id2label=dataset_config.id2label,
    )

    return itr_scores


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
        [
            token_id
            for token_id in completion_ids
            if token_id not in [tokenizer.bos_token_id, tokenizer.eos_token_id]
        ]
        for completion_ids in generated_sequences
    ]

    # For some models the generated tokens also includes the input tokens, so we need
    # to deal with both cases when extracting the predicted labels
    prompt_prefix_exists = dataset_config.prompt_prefix != ""
    pred_idx = dataset_config.num_few_shot_examples + int(prompt_prefix_exists)
    raw_predictions: list[str] = list()
    for completion_ids_list in completion_ids_lists:
        decoded = tokenizer.decode(completion_ids_list)
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
