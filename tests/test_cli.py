"""Unit tests for the `cli` module."""

import pytest
from click.types import BOOL, STRING, Choice

from scandeval.cli import benchmark


@pytest.fixture(scope="module")
def params():
    ctx = benchmark.make_context(None, list())
    yield {p.name: p.type for p in benchmark.get_params(ctx)}


def test_cli_param_names(params):
    assert set(params.keys()) == {
        "model_id",
        "dataset",
        "language",
        "model_language",
        "model_framework",
        "dataset_language",
        "dataset_task",
        "batch_size",
        "evaluate_train",
        "progress_bar",
        "raise_errors",
        "verbose",
        "save_results",
        "cache_dir",
        "auth_token",
        "use_auth_token",
        "ignore_duplicates",
        "help",
    }


def test_cli_param_types(params):
    assert params["model_id"] == STRING
    assert isinstance(params["dataset"], Choice)
    assert isinstance(params["language"], Choice)
    assert isinstance(params["model_language"], Choice)
    assert isinstance(params["dataset_language"], Choice)
    assert isinstance(params["dataset_task"], Choice)
    assert isinstance(params["batch_size"], Choice)
    assert params["evaluate_train"] == BOOL
    assert params["progress_bar"] == BOOL
    assert params["raise_errors"] == BOOL
    assert params["verbose"] == BOOL
    assert params["save_results"] == BOOL
    assert params["cache_dir"] == STRING
    assert params["auth_token"] == STRING
    assert params["use_auth_token"] == BOOL
    assert params["ignore_duplicates"] == BOOL
    assert params["help"] == BOOL
