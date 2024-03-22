"""Factory which produces datasets from a configuration."""

from typing import TYPE_CHECKING, Type

from .dataset_configs import get_dataset_config
from .utils import get_class_by_name

if TYPE_CHECKING:
    from .benchmark_dataset import BenchmarkDataset
    from .config import BenchmarkConfig, DatasetConfig


class DatasetFactory:
    """Factory which produces datasets from a configuration.

    Attributes:
        benchmark_config:
            The benchmark configuration to be used in all datasets constructed.
    """

    def __init__(self, benchmark_config: "BenchmarkConfig") -> None:
        """Initialize the dataset factory.

        Args:
            benchmark_config:
                The benchmark configuration to be used in all datasets constructed.
        """
        self.benchmark_config = benchmark_config

    def build_dataset(self, dataset: "str | DatasetConfig") -> "BenchmarkDataset":
        """Build a dataset from a configuration or a name.

        Args:
            dataset:
                The name of the dataset, or the dataset configuration.

        Returns:
            BenchmarkDataset:
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
        benchmark_cls: Type["BenchmarkDataset"] | None = get_class_by_name(
            class_name=potential_class_names
        )
        if not benchmark_cls:
            raise ValueError(
                "Could not find a benchmark class for any of the following potential "
                f"names: {', '.join(potential_class_names)}."
            )

        # Create the dataset
        dataset_obj = benchmark_cls(
            dataset_config=dataset_config, benchmark_config=self.benchmark_config
        )

        return dataset_obj
