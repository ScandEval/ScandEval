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
        "dataset_language",
        "model_task",
        "dataset_task",
        "evaluate_train",
        "no_progress_bar",
        "raise_error_on_invalid_model",
        "verbose",
        "no_save_results",
        "cache_dir",
        "auth_token",
        "use_auth_token",
        "help",
    }


def test_cli_param_types(params):
    assert params["model_id"] == STRING
    assert isinstance(params["dataset"], Choice)
    assert isinstance(params["language"], Choice)
    assert isinstance(params["model_language"], Choice)
    assert isinstance(params["dataset_language"], Choice)
    assert params["model_task"] == STRING
    assert isinstance(params["dataset_task"], Choice)
    assert params["evaluate_train"] == BOOL
    assert params["no_progress_bar"] == BOOL
    assert params["raise_error_on_invalid_model"] == BOOL
    assert params["verbose"] == BOOL
    assert params["no_save_results"] == BOOL
    assert params["cache_dir"] == STRING
    assert params["auth_token"] == STRING
    assert params["use_auth_token"] == BOOL
    assert params["help"] == BOOL
