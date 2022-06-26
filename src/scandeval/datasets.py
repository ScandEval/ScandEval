"""Functions related to loading of datasets."""

from pathlib import Path
from typing import Sequence

import yaml
from pkg_resources import resource_filename

from .config import DatasetConfig


def get_config_dir() -> Path:
    """Get the path to the config directory.

    Returns:
        Path:
            The path to the config directory.
    """
    return Path(resource_filename("scandeval", "configs"))


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get the dataset configuration for a dataset.

    Args:
        dataset_name (str):
            The name of the dataset.

    Returns:
        DatasetConfig:
            The dataset configuration.
    """
    # Get the path to the dataset configuration file
    dataset_config_path = get_config_dir() / "datasets" / f"{dataset_name}.yaml"

    # Load the YAML dataset configuration
    with dataset_config_path.open() as f:
        dataset_config_dict = yaml.safe_load(f)

    # Extract the task from the dataset configuration and remove it from the
    # dataset configuration as well
    task = dataset_config_dict["task"]

    # Load the task configuration
    task_config_path = get_config_dir() / "tasks" / f"{task}.yaml"
    with task_config_path.open() as f:
        task_config_dict = yaml.safe_load(f)

        # Remove the `name` attribute, as we won't need that here
        task_config_dict.pop("name")

    # Merge the two configurations
    dataset_config_dict = {**dataset_config_dict, **task_config_dict}

    # Add the `label_synonyms` attribute
    dataset_config_dict["label_synonyms"] = [
        [main_lbl] + other_lbls
        for main_lbl, other_lbls in dataset_config_dict["labels"].items()
    ]

    # Add the `id2label` and `num_labels` attributes
    dataset_config_dict["id2label"] = list(dataset_config_dict["labels"].keys())
    dataset_config_dict["num_labels"] = len(dataset_config_dict["id2label"])

    # Add the `label2id` attribute
    dataset_config_dict["label2id"] = {
        lbl: idx
        for idx, synset in enumerate(dataset_config_dict["label_synonyms"])
        for lbl in synset
    }

    # Remove the `labels` attribute
    dataset_config_dict.pop("labels")

    # Create and return the dataset configuration
    return DatasetConfig(**dataset_config_dict)


def get_all_dataset_configs() -> Sequence[DatasetConfig]:
    """Get a list of all the datasets.

    Returns:
        list:
            A list of all the datasets.
    """
    datasets_config_dir = get_config_dir() / "datasets"
    datasets = list()
    for dataset_config_path in datasets_config_dir.glob("*.yaml"):
        dataset_config = get_dataset_config(dataset_config_path.stem)
        datasets.append(dataset_config)
    return datasets
