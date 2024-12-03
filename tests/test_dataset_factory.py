"""Unit tests for the `dataset_factory` module."""

from copy import deepcopy

from scandeval.benchmark_datasets import (
    NamedEntityRecognition,
    QuestionAnswering,
    SequenceClassification,
    TextToText,
)
from scandeval.dataset_factory import build_benchmark_dataset
from scandeval.tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SUMM


def test_configs_are_preserved(dataset_config, benchmark_config):
    """Test that the configs are preserved."""
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert dataset.benchmark_config == benchmark_config
    assert dataset.dataset_config == dataset_config


def test_build_sent_dataset(dataset_config, benchmark_config):
    """Test that SENT datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = SENT
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, SequenceClassification)


def test_build_la_dataset(dataset_config, benchmark_config):
    """Test that LA datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = LA
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, SequenceClassification)


def test_build_ner_dataset(dataset_config, benchmark_config):
    """Test that NER datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = NER
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, NamedEntityRecognition)


def test_build_qa_dataset(dataset_config, benchmark_config):
    """Test that RC datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = RC
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, QuestionAnswering)


def test_build_summ_dataset(dataset_config, benchmark_config):
    """Test that SUMM datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = SUMM
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, TextToText)


def test_build_know_dataset(dataset_config, benchmark_config):
    """Test that KNOW datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = KNOW
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, SequenceClassification)


def test_build_common_sense_dataset(dataset_config, benchmark_config):
    """Test that COMMON_SENSE datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = COMMON_SENSE
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, SequenceClassification)


def test_build_mcrc_dataset(dataset_config, benchmark_config):
    """Test that MCRC datasets are built correctly."""
    cfg = deepcopy(dataset_config)
    cfg.task = MCRC
    dataset = build_benchmark_dataset(
        dataset=dataset_config, benchmark_config=benchmark_config
    )
    assert isinstance(dataset, SequenceClassification)
