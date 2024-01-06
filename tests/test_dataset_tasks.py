"""Unit tests for the `dataset_tasks` module."""

from typing import Generator

import pytest
from scandeval.config import DatasetTask
from scandeval.dataset_tasks import get_all_dataset_tasks


class TestGetAllDatasetTasks:
    """Unit tests for the `get_all_dataset_tasks` function."""

    @pytest.fixture(scope="class")
    def dataset_tasks(self) -> Generator[dict[str, DatasetTask], None, None]:
        """Yields all dataset tasks."""
        yield get_all_dataset_tasks()

    def test_dataset_tasks_is_dict(self, dataset_tasks):
        """Tests that the dataset tasks are a dictionary."""
        assert isinstance(dataset_tasks, dict)

    def test_dataset_tasks_are_objects(self, dataset_tasks):
        """Tests that the dataset tasks are objects."""
        for dataset_task in dataset_tasks.values():
            assert isinstance(dataset_task, DatasetTask)

    @pytest.mark.parametrize(
        "dataset_task_name",
        [
            "linguistic-acceptability",
            "named-entity-recognition",
            "question-answering",
            "sentiment-classification",
            "summarization",
            "knowledge",
            "common-sense-reasoning",
            "text-modelling",
            "speed",
        ],
    )
    def test_get_task(self, dataset_tasks, dataset_task_name):
        """Tests that the dataset task can be retrieved by name."""
        assert dataset_task_name in dataset_tasks
