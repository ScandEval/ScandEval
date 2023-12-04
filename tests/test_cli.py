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
        "dataset_task",
        "batch_size",
        "evaluate_train",
        "progress_bar",
        "raise_errors",
        "verbose",
        "save_results",
        "cache_dir",
        "token",
        "use_token",
        "ignore_duplicates",
        "framework",
        "device",
        "trust_remote_code",
        "load_in_4bit",
        "use_flash_attention",
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
    assert params["token"] == STRING
    assert params["use_token"] == BOOL
    assert params["ignore_duplicates"] == BOOL
    assert isinstance(params["framework"], Choice)
    assert isinstance(params["device"], Choice)
    assert params["trust_remote_code"] == BOOL
    assert params["load_in_4bit"] == BOOL
    assert params["use_flash_attention"] == BOOL
    assert params["help"] == BOOL
