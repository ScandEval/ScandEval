"""General fixtures used throughout test modules."""

import os
import sys
from typing import Generator

import pytest
import torch
from scandeval.config import BenchmarkConfig
from scandeval.dataset_configs import MMLU_DA_CONFIG, SPEED_CONFIG
from scandeval.dataset_tasks import LA, NER, QA, SENT
from scandeval.languages import DA, NO, SV


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


@pytest.fixture(scope="session")
def benchmark_config() -> Generator[BenchmarkConfig, None, None]:
    # Get the authentication token to the Hugging Face Hub
    auth = os.environ.get("HUGGINGFACE_HUB_TOKEN", True)

    # Ensure that the token does not contain quotes or whitespace
    if isinstance(auth, str):
        auth = auth.strip(" \"'")

    yield BenchmarkConfig(
        model_languages=[DA, SV, NO],
        dataset_languages=[DA, SV, NO],
        dataset_tasks=[NER, QA, SENT, LA],
        framework=None,
        batch_size=32,
        raise_errors=False,
        cache_dir=".scandeval_cache",
        evaluate_train=False,
        token=auth,
        openai_api_key=None,
        progress_bar=False,
        save_results=False,
        device=torch.device("cpu"),
        verbose=False,
        trust_remote_code=True,
        load_in_4bit=None,
        use_flash_attention=False,
        clear_model_cache=False,
        only_validation_split=False,
    )


@pytest.fixture(scope="session")
def model_id():
    yield "jonfd/electra-small-nordic"


@pytest.fixture(scope="session")
def generative_model_id():
    yield "AI-Sweden-Models/gpt-sw3-126m"


@pytest.fixture(scope="session")
def dataset_config():
    yield SPEED_CONFIG


@pytest.fixture(scope="session")
def generative_dataset_config():
    yield MMLU_DA_CONFIG
