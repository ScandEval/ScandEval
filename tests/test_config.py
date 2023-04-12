"""Unit tests for the `config` module."""

import pytest

from scandeval.config import (
    BenchmarkConfig,
    DatasetConfig,
    DatasetTask,
    Language,
    MetricConfig,
    ModelConfig,
)


@pytest.fixture(scope="module")
def metric_config():
    yield MetricConfig(
        name="metric_name",
        pretty_name="Metric name",
        huggingface_id="metric_id",
        results_key="metric_key",
    )


@pytest.fixture(scope="class")
def dataset_task(metric_config):
    yield DatasetTask(
        name="dataset_task_name",
        supertask="supertask_name",
        metrics=[metric_config],
        labels=["label"],
    )


@pytest.fixture(scope="class")
def language():
    yield Language(code="language_code", name="Language name")


class TestMetricConfig:
    def test_metric_config_is_object(self, metric_config):
        assert isinstance(metric_config, MetricConfig)

    def test_attributes_correspond_to_arguments(self, metric_config):
        assert metric_config.name == "metric_name"
        assert metric_config.pretty_name == "Metric name"
        assert metric_config.huggingface_id == "metric_id"
        assert metric_config.results_key == "metric_key"

    def test_default_value_of_compute_kwargs(self, metric_config):
        assert metric_config.compute_kwargs == dict()


class TestDatasetTask:
    def test_dataset_task_is_object(self, dataset_task):
        assert isinstance(dataset_task, DatasetTask)

    def test_attributes_correspond_to_arguments(self, dataset_task, metric_config):
        assert dataset_task.name == "dataset_task_name"
        assert dataset_task.supertask == "supertask_name"
        assert dataset_task.metrics == [metric_config]
        assert dataset_task.labels == ["label"]


class TestLanguage:
    def test_language_is_object(self, language):
        assert isinstance(language, Language)

    def test_attributes_correspond_to_arguments(self, language):
        assert language.code == "language_code"
        assert language.name == "Language name"


class TestBenchmarkConfig:
    @pytest.fixture(scope="class")
    def benchmark_config(self, language, dataset_task):
        yield BenchmarkConfig(
            model_languages=[language],
            dataset_languages=[language],
            dataset_tasks=[dataset_task],
            raise_errors=True,
            cache_dir="cache_dir",
            evaluate_train=True,
            use_auth_token=True,
            progress_bar=True,
            save_results=True,
            verbose=True,
            batch_size=32,
        )

    def test_benchmark_config_is_object(self, benchmark_config):
        assert isinstance(benchmark_config, BenchmarkConfig)

    def test_attributes_correspond_to_arguments(
        self, benchmark_config, language, dataset_task
    ):
        assert benchmark_config.model_languages == [language]
        assert benchmark_config.dataset_languages == [language]
        assert benchmark_config.dataset_tasks == [dataset_task]
        assert benchmark_config.raise_errors is True
        assert benchmark_config.cache_dir == "cache_dir"
        assert benchmark_config.evaluate_train is True
        assert benchmark_config.use_auth_token is True
        assert benchmark_config.progress_bar is True
        assert benchmark_config.save_results is True
        assert benchmark_config.verbose is True


class TestDatasetConfig:
    @pytest.fixture(scope="class")
    def dataset_config(self, language, dataset_task):
        yield DatasetConfig(
            name="dataset_name",
            pretty_name="Dataset name",
            huggingface_id="dataset_id",
            task=dataset_task,
            languages=[language],
        )

    def test_dataset_config_is_object(self, dataset_config):
        assert isinstance(dataset_config, DatasetConfig)

    def test_attributes_correspond_to_arguments(
        self, dataset_config, language, dataset_task
    ):
        assert dataset_config.name == "dataset_name"
        assert dataset_config.pretty_name == "Dataset name"
        assert dataset_config.huggingface_id == "dataset_id"
        assert dataset_config.task == dataset_task
        assert dataset_config.languages == [language]

    def test_id2label(self, dataset_config):
        assert dataset_config.id2label == ["label"]

    def test_label2id(self, dataset_config):
        assert dataset_config.label2id == dict(label=0)

    def test_num_labels(self, dataset_config):
        assert dataset_config.num_labels == 1


class TestModelConfig:
    @pytest.fixture(scope="class")
    def model_config(self, language):
        yield ModelConfig(
            model_id="model_id",
            revision="revision",
            framework="framework",
            task="task",
            languages=[language],
        )

    def test_model_config_is_object(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_attributes_correspond_to_arguments(self, model_config, language):
        assert model_config.model_id == "model_id"
        assert model_config.revision == "revision"
        assert model_config.framework == "framework"
        assert model_config.task == "task"
        assert model_config.languages == [language]
