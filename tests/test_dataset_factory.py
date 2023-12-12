"""Unit tests for the `dataset_factory` module."""

from copy import deepcopy

import pytest
from scandeval.config import DatasetConfig
from scandeval.dataset_factory import DatasetFactory
from scandeval.dataset_tasks import LA, NER, QA, SENT
from scandeval.languages import DA, IS
from scandeval.named_entity_recognition import NamedEntityRecognition
from scandeval.question_answering import QuestionAnswering
from scandeval.sequence_classification import SequenceClassification


@pytest.fixture(scope="module")
def dataset_config():
    yield DatasetConfig(
        name="test_dataset",
        pretty_name="Test Dataset",
        huggingface_id="test_dataset",
        task=SENT,
        languages=[DA, IS],
        prompt_template="{text}\n{label}",
        max_generated_tokens=1,
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
    assert isinstance(dataset, SequenceClassification)


def test_build_la_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = LA
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, SequenceClassification)


def test_build_ner_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = NER
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, NamedEntityRecognition)


def test_build_qa_dataset(dataset_factory, dataset_config):
    cfg = deepcopy(dataset_config)
    cfg.task = QA
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, QuestionAnswering)
