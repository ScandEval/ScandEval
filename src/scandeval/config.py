"""Configuration classes used throughout the project."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass
class BenchmarkConfig:
    """General benchmarking configuration, across datasets and models.

    Attributes:
        languages (list of str):
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this to
            'all' if all languages (also non-Scandinavian) should be considered.
        model_languages (list of str or None):
            The language codes of the languages to include for models. If not None
            then this overrides the `language` parameter for model languages.
        dataset_languages (list of str or None):
            The language codes of the languages to include for datasets. If not None
            then this overrides the `language` parameter for dataset languages.
        tasks (list of str):
            The tasks to consider in the list. If 'all' then all tasks will be
            considered.
        raise_error_on_invalid_model (bool):
            Whether to raise an error if a model is invalid.
        cache_dir (str):
            Directory to store cached models and datasets.
        evaluate_train (bool):
            Whether to evaluate on the training set.
        use_auth_token (bool):
            Whether to use an authentication token.
        progress_bar (bool):
            Whether to show a progress bar.
        save_results (bool):
            Whether to save the benchmark results to
            'scandeval_benchmark_results.json'.
        verbose (bool):
            Whether to print verbose output.
    """

    languages: Sequence[Optional[str]]
    model_languages: Sequence[Optional[str]]
    dataset_languages: Sequence[Optional[str]]
    tasks: Sequence[Optional[str]]
    raise_error_on_invalid_model: bool
    cache_dir: str
    evaluate_train: bool
    use_auth_token: bool
    progress_bar: bool
    save_results: bool
    verbose: bool


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
        task (str):
            The task of the dataset.
        supertask (str):
            The supertask of the dataset.
        languages (list of str):
            The ISO 639-1 language codes of the entries in the dataset.
        num_labels (int):
            The number of labels in the dataset.
        id2label (list of str):
            The mapping from IDs to labels.
        label2id (dict)
            The mapping from labels to IDs.
        label_synonyms (list of list of str):
            The synonyms of labels.
        metrics (dict):
            A mapping from metric keys to their configuration.
    """

    name: str
    pretty_name: str
    huggingface_id: str
    task: str
    supertask: str
    languages: Sequence[str]
    num_labels: int
    id2label: Sequence[str]
    label2id: Dict[str, int]
    label_synonyms: Sequence[Sequence[str]]
    metrics: Dict[str, Dict[str, str]]


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        model_id (str):
            The ID of the model.
        framework (str):
            The framework of the model.
        task (str):
            The task of the model.
        languages (list of str):
            The ISO 639-1 language codes of the model.
        revision (str):
            The revision of the model.
    """

    model_id: str
    framework: str
    task: str
    languages: Sequence[str]
    revision: str
