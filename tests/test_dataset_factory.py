"""Unit tests for the `dataset_factory` module."""

from copy import deepcopy

import pytest

from src.scandeval.config import BenchmarkConfig, DatasetConfig
from src.scandeval.dataset_factory import DatasetFactory
from src.scandeval.dataset_tasks import LA, NER, QA, SENT
from src.scandeval.languages import DA, FO, IS, NO, SV
from src.scandeval.ner import NERBenchmark
from src.scandeval.qa import QABenchmark
from src.scandeval.text_classification import TextClassificationBenchmark


@pytest.fixture(scope="module")
def benchmark_config():
    yield BenchmarkConfig(
        model_languages=[DA, SV, NO],
        dataset_languages=[IS, FO],
        model_tasks=None,
        dataset_tasks=[SENT, NER, LA, QA],
        raise_error_on_invalid_model=False,
        cache_dir=".",
        evaluate_train=True,
        use_auth_token=False,
        progress_bar=False,
        save_results=False,
        verbose=False,
    )


@pytest.fixture(scope="module")
def dataset_config():
    yield DatasetConfig(
        name="test_dataset",
        pretty_name="Test Dataset",
        huggingface_id="test_dataset",
        task=SENT,
        languages=[DA, IS],
    )


@pytest.fixture(scope="module")
def dataset_factory(benchmark_config):
    yield DatasetFactory(benchmark_config)


def test_attributes_correspond_to_arguments(dataset_factory, benchmark_config):
    assert dataset_factory.benchmark_config == benchmark_config


def test_configs_are_preserved(dataset_factory, benchmark_config, dataset_config):
    dataset = dataset_factory.build_dataset(dataset=dataset_config)
    assert dataset.benchmark_config == benchmark_config
    assert dataset.dataset_config == dataset_config


def test_build_sent_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = SENT
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, TextClassificationBenchmark)


def test_build_la_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = LA
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, TextClassificationBenchmark)


def test_build_ner_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = NER
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, NERBenchmark)


def test_build_qa_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = QA
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, QABenchmark)
