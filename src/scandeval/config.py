"""Configuration classes used throughout the project."""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .enums import Framework, ModelType


@dataclass
class MetricConfig:
    """Configuration for a metric.

    Attributes:
        name (str):
            The name of the metric.
        pretty_name (str):
            A longer prettier name for the metric, which allows cases and spaces. Used
            for logging.
        huggingface_id (str):
            The Hugging Face ID of the metric.
        results_key (str):
            The name of the key used to extract the metric scores from the results
            dictionary.
        compute_kwargs (dict, optional):
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
        postprocessing_fn (callable, optional):
            A function to apply to the metric scores after they are computed, taking
            the score to the postprocessed score along with its string representation.
            Defaults to x -> (100 * x, f"{x:.2%}").
    """

    name: str
    pretty_name: str
    huggingface_id: str
    results_key: str
    compute_kwargs: dict[str, Any] = field(default_factory=dict)
    postprocessing_fn: Callable[[float], tuple[float, str]] = field(
        default_factory=lambda: lambda raw_score: (100 * raw_score, f"{raw_score:.2%}")
    )


@dataclass
class DatasetTask:
    """A dataset task.

    Attributes:
        name (str):
            The name of the task.
        supertask (str):
            The supertask of the task, describing the overall type of task.
        metrics (list of MetricConfig objects):
            The metrics used to evaluate the task.
        labels (list of str):
            The labels used in the task.
    """

    name: str
    supertask: str
    metrics: list[MetricConfig]
    labels: list[str]


@dataclass
class Language:
    """A benchmarkable language.

    Attributes:
        code (str):
            The ISO 639-1 language code of the language.
        name (str):
            The name of the language.
    """

    code: str
    name: str


@dataclass
class BenchmarkConfig:
    """General benchmarking configuration, across datasets and models.

    Attributes:
        model_languages (list of Language objects):
            The languages of the models to benchmark.
        framework (Framework or None):
            The framework of the models to benchmark. If None then the framework will
            be inferred.
        few_shot (bool):
            Whether to benchmark the models in few-shot mode.
        dataset_languages (list of Language objects):
            The languages of the datasets in the benchmark.
        dataset_tasks (list of DatasetTask):
            The tasks to benchmark.
        batch_size (int):
            The batch size to use.
        raise_errors (bool):
            Whether to raise errors instead of skipping them.
        cache_dir (str):
            Directory to store cached models and datasets.
        evaluate_train (bool):
            Whether to evaluate on the training set.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        openai_api_key (str or None):
            The API key for the OpenAI API. If None then OpenAI models will not be
            benchmarked.
        progress_bar (bool):
            Whether to show a progress bar.
        save_results (bool):
            Whether to save the benchmark results to
            'scandeval_benchmark_results.json'.
        device (torch.device):
            The device to use for benchmarking.
        verbose (bool):
            Whether to print verbose output.
        testing (bool, optional):
            Whether a unit test is being run. Defaults to False.
    """

    model_languages: list[Language]
    framework: Framework | str | None
    few_shot: bool
    dataset_languages: list[Language]
    dataset_tasks: list[DatasetTask]
    batch_size: int
    raise_errors: bool
    cache_dir: str
    evaluate_train: bool
    use_auth_token: bool | str
    openai_api_key: str | None
    progress_bar: bool
    save_results: bool
    device: torch.device
    verbose: bool
    testing: bool = False


@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        name (str):
            The name of the dataset. Must be lower case with no spaces.
        pretty_name (str):
            A longer prettier name for the dataset, which allows cases and spaces. Used
            for logging.
        huggingface_id (str):
            The Hugging Face ID of the dataset.
        task (DatasetTask):
            The task of the dataset.
        languages (sequence of Language objects):
            The ISO 639-1 language codes of the entries in the dataset.
        id2label (list of str):
            The mapping from ID to label.
        label2id (dict of str to int):
            The mapping from label to ID.
        num_labels (int):
            The number of labels in the dataset.
        prompt_template (str):
            The template for the prompt to use when benchmarking the dataset using
            few-shot evaluation.
        num_few_shot_examples (int):
            The number of examples to use when benchmarking the dataset using few-shot
            evaluation. For a classification task, these will be drawn evenly from
            each label.
    """

    name: str
    pretty_name: str
    huggingface_id: str
    task: DatasetTask
    languages: list[Language]
    prompt_template: str
    num_few_shot_examples: int

    @property
    def id2label(self) -> list[str]:
        return [label for label in self.task.labels]

    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.task.labels)}

    @property
    def num_labels(self) -> int:
        return len(self.task.labels)


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        model_id (str):
            The ID of the model.
        revision (str):
            The revision of the model.
        framework (Framework):
            The framework of the model.
        task (str):
            The task that the model was trained on.
        languages (sequence of Language objects):
            The languages of the model.
        model_type (ModelType):
            The type of the model.
    """

    model_id: str
    revision: str
    framework: Framework | str
    task: str
    languages: list[Language]
    model_type: ModelType | str
