"""Unit tests for the `config` module."""

import pytest

from scandeval.config import (
    BenchmarkConfig,
    DatasetConfig,
    Language,
    MetricConfig,
    ModelConfig,
    Task,
)
from scandeval.enums import Framework, ModelType


class TestMetricConfig:
    """Unit tests for the `MetricConfig` class."""

    def test_metric_config_is_object(self, metric_config):
        """Test that the metric config is a `MetricConfig` object."""
        assert isinstance(metric_config, MetricConfig)

    def test_attributes_correspond_to_arguments(self, metric_config):
        """Test that the metric config attributes correspond to the arguments."""
        assert metric_config.name == "metric_name"
        assert metric_config.pretty_name == "Metric name"
        assert metric_config.huggingface_id == "metric_id"
        assert metric_config.results_key == "metric_key"

    def test_default_value_of_compute_kwargs(self, metric_config):
        """Test that the default value of `compute_kwargs` is an empty dictionary."""
        assert metric_config.compute_kwargs == dict()

    @pytest.mark.parametrize(
        "inputs,expected",
        [
            (0.5, (50.0, "50.00%")),
            (0.123456, (12.3456, "12.35%")),
            (0.0, (0.0, "0.00%")),
            (1.0, (100.0, "100.00%")),
            (0.999999, (99.9999, "100.00%")),
            (2.0, (200.0, "200.00%")),
            (-1.0, (-100.0, "-100.00%")),
        ],
    )
    def test_default_value_of_postprocessing_fn(self, metric_config, inputs, expected):
        """Test that the default value of `postprocessing_fn` is correct."""
        assert metric_config.postprocessing_fn(inputs) == expected


class TestTask:
    """Unit tests for the `Task` class."""

    def test_task_is_object(self, task):
        """Test that the dataset task is a `Task` object."""
        assert isinstance(task, Task)

    def test_attributes_correspond_to_arguments(self, task):
        """Test that the dataset task attributes correspond to the arguments."""
        assert task.name == "speed"
        assert task.supertask == "sequence-classification"
        assert task.labels == []


class TestLanguage:
    """Unit tests for the `Language` class."""

    def test_language_is_object(self, language):
        """Test that the language is a `Language` object."""
        assert isinstance(language, Language)

    def test_attributes_correspond_to_arguments(self, language):
        """Test that the language attributes correspond to the arguments."""
        assert language.code == "language_code"
        assert language.name == "Language name"


class TestBenchmarkConfig:
    """Unit tests for the `BenchmarkConfig` class."""

    def test_benchmark_config_is_object(self, benchmark_config):
        """Test that the benchmark config is a `BenchmarkConfig` object."""
        assert isinstance(benchmark_config, BenchmarkConfig)

    def test_attributes_correspond_to_arguments(
        self, benchmark_config, language, task, auth, device
    ):
        """Test that the benchmark config attributes correspond to the arguments."""
        assert benchmark_config.model_languages == [language]
        assert benchmark_config.dataset_languages == [language]
        assert benchmark_config.tasks == [task]
        assert benchmark_config.framework is None
        assert benchmark_config.batch_size == 32
        assert benchmark_config.raise_errors is False
        assert benchmark_config.cache_dir == ".scandeval_cache"
        assert benchmark_config.evaluate_train is False
        assert benchmark_config.token is auth
        assert benchmark_config.openai_api_key is None
        assert benchmark_config.progress_bar is False
        assert benchmark_config.save_results is True
        assert benchmark_config.device == device
        assert benchmark_config.verbose is False
        assert benchmark_config.trust_remote_code is True
        assert benchmark_config.load_in_4bit is None
        assert benchmark_config.use_flash_attention is False
        assert benchmark_config.clear_model_cache is False
        assert benchmark_config.only_validation_split is False
        assert benchmark_config.few_shot is True


class TestDatasetConfig:
    """Unit tests for the `DatasetConfig` class."""

    def test_dataset_config_is_object(self, dataset_config):
        """Test that the dataset config is a `DatasetConfig` object."""
        assert isinstance(dataset_config, DatasetConfig)

    def test_attributes_correspond_to_arguments(self, dataset_config, language, task):
        """Test that the dataset config attributes correspond to the arguments."""
        assert dataset_config.name == "dataset_name"
        assert dataset_config.pretty_name == "Dataset name"
        assert dataset_config.huggingface_id == "dataset_id"
        assert dataset_config.task == task
        assert dataset_config.languages == [language]
        assert dataset_config.prompt_template == "{text}\n{label}"
        assert dataset_config.max_generated_tokens == 1

    def test_id2label(self, dataset_config):
        """Test that the `id2label` attribute is correct."""
        assert dataset_config.id2label == dict()

    def test_label2id(self, dataset_config):
        """Test that the `label2id` attribute is correct."""
        assert dataset_config.label2id == dict()

    def test_num_labels(self, dataset_config):
        """Test that the `num_labels` attribute is correct."""
        assert dataset_config.num_labels == 0

    def test_default_value_of_prompt_label_mapping(self, dataset_config):
        """Test that the default value of `prompt_label_mapping` is an empty dictionary."""
        assert dataset_config.prompt_label_mapping == dict()


class TestModelConfig:
    """Unit tests for the `ModelConfig` class."""

    def test_model_config_is_object(self, model_config):
        """Test that the model config is a `ModelConfig` object."""
        assert isinstance(model_config, ModelConfig)

    def test_attributes_correspond_to_arguments(self, model_config, language):
        """Test that the model config attributes correspond to the arguments."""
        assert model_config.model_id == "model_id"
        assert model_config.revision == "revision"
        assert model_config.framework == Framework.PYTORCH
        assert model_config.task == "task"
        assert model_config.languages == [language]
        assert model_config.model_type == ModelType.FRESH
        assert model_config.model_cache_dir == "cache_dir"
