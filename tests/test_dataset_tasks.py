"""Unit tests for the `dataset_tasks` module."""

import pytest

from scandeval.config import DatasetTask
from scandeval.dataset_tasks import get_all_dataset_tasks


class TestGetAllDatasetTasks:
    @pytest.fixture(scope="class")
    def dataset_tasks(self):
        yield get_all_dataset_tasks()

    def test_dataset_tasks_is_dict(self, dataset_tasks):
        assert isinstance(dataset_tasks, dict)

    def test_dataset_tasks_are_objects(self, dataset_tasks):
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
            "speed",
        ],
    )
    def test_get_task(self, dataset_tasks, dataset_task_name):
        assert dataset_task_name in dataset_tasks
