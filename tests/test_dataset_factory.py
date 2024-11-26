"""Unit tests for the `dataset_factory` module."""

from copy import deepcopy
from typing import Generator

import pytest

from scandeval.dataset_factory import DatasetFactory
from scandeval.named_entity_recognition import NamedEntityRecognition
from scandeval.question_answering import QuestionAnswering
from scandeval.sequence_classification import SequenceClassification
from scandeval.tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM
from scandeval.text_to_text import TextToText


@pytest.fixture(scope="module")
def dataset_factory(benchmark_config) -> Generator[DatasetFactory, None, None]:
    """Yields a dataset factory."""
    yield DatasetFactory(benchmark_config)


def test_attributes_correspond_to_arguments(dataset_factory, benchmark_config):
    """Test that the attributes correspond to the arguments."""
    assert dataset_factory.benchmark_config == benchmark_config


def test_configs_are_preserved(dataset_factory, benchmark_config, dataset_config):
    """Test that the configs are preserved."""
    dataset = dataset_factory.build_dataset(dataset=dataset_config)
    assert dataset.benchmark_config == benchmark_config
    assert dataset.dataset_config == dataset_config


def test_build_sent_dataset(dataset_factory, dataset_config):
    """Test that SENT datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = SENT
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, SequenceClassification)


def test_build_la_dataset(dataset_factory, dataset_config):
    """Test that LA datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = LA
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, SequenceClassification)


def test_build_ner_dataset(dataset_factory, dataset_config):
    """Test that NER datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = NER
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, NamedEntityRecognition)


def test_build_qa_dataset(dataset_factory, dataset_config):
    """Test that RC datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = RC
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, QuestionAnswering)


def test_build_summ_dataset(dataset_factory, dataset_config):
    """Test that SUMM datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = SUMM
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, TextToText)


def test_build_know_dataset(dataset_factory, dataset_config):
    """Test that KNOW datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = KNOW
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, SequenceClassification)


def test_build_common_sense_dataset(dataset_factory, dataset_config):
    """Test that COMMON_SENSE datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = COMMON_SENSE
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, SequenceClassification)


def test_build_mcrc_dataset(dataset_factory, dataset_config):
    """Test that MCRC datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = MCRC
    dataset = dataset_factory.build_dataset(dataset=cfg)
    assert isinstance(dataset, SequenceClassification)


@pytest.mark.skip(reason="Text modelling datasets are not yet implemented.")
def test_build_text_modelling_dataset(dataset_factory, dataset_config):
    """Test that TEXT_MODELLING datasets are built correctly."""
    pass
