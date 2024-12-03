"""Factory which produces datasets from a configuration."""

import typing as t

from .dataset_configs import get_dataset_config
from .utils import get_class_by_name

if t.TYPE_CHECKING:
    from .benchmark_datasets import BenchmarkDataset
    from .data_models import BenchmarkConfig, DatasetConfig


def build_benchmark_dataset(
    dataset: "str | DatasetConfig", benchmark_config: "BenchmarkConfig"
) -> "BenchmarkDataset":
    """Build a benchmark dataset from a configuration or a name.

    Args:
        dataset:
            The name of the dataset, or the dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The benchmark dataset.
    """
    # Get the dataset configuration
    if isinstance(dataset, str):
        dataset_config = get_dataset_config(dataset)
    else:
        dataset_config = dataset

    # Get the benchmark class based on the task
    potential_class_names = [
        dataset_config.name,
        dataset_config.task.name,
        dataset_config.task.supertask,
    ]
    benchmark_cls: t.Type["BenchmarkDataset"] | None = get_class_by_name(
        class_name=potential_class_names
    )
    if not benchmark_cls:
        raise ValueError(
            "Could not find a benchmark class for any of the following potential "
            f"names: {', '.join(potential_class_names)}."
        )

    # Create the dataset
    dataset_obj = benchmark_cls(
        dataset_config=dataset_config, benchmark_config=benchmark_config
    )

    return dataset_obj
