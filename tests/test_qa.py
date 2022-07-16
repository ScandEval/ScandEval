"""Unit tests for the `qa` module."""

import pytest

from src.scandeval.config import BenchmarkConfig
from src.scandeval.dataset_tasks import QA
from src.scandeval.languages import DA, FO, IS, NO, SV
from src.scandeval.qa import QABenchmark


@pytest.fixture(scope="module")
def benchmark_config():
    yield BenchmarkConfig(
        model_languages=[DA, SV, NO, IS, FO],
        dataset_languages=[DA, SV, NO, IS, FO],
        model_tasks=None,
        dataset_tasks=[QA],
        raise_error_on_invalid_model=False,
        cache_dir=".scandeval_cache",
        evaluate_train=True,
        use_auth_token=False,
        progress_bar=True,
        save_results=False,
        verbose=True,
        testing=True,
    )


@pytest.fixture(scope="module")
def model_id():
    yield "Maltehb/aelaectra-danish-electra-small-cased"
