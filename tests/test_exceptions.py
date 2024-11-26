"""Unit tests for the `exceptions` module."""

import pytest

from scandeval.exceptions import (
    FlashAttentionNotInstalled,
    HuggingFaceHubDown,
    InvalidBenchmark,
    InvalidModel,
    MissingHuggingFaceToken,
    NaNValueInModelOutput,
    NeedsAdditionalArgument,
    NeedsExtraInstalled,
    NoInternetConnection,
)


def test_invalid_benchmark_is_an_exception():
    """Test that `InvalidBenchmark` is an exception."""
    with pytest.raises(InvalidBenchmark):
        raise InvalidBenchmark()


def test_invalid_model_is_an_exception():
    """Test that `InvalidModel` is an exception."""
    with pytest.raises(InvalidModel):
        raise InvalidModel()


def test_hugging_face_hub_down_is_an_exception():
    """Test that `HuggingFaceHubDown` is an exception."""
    with pytest.raises(HuggingFaceHubDown):
        raise HuggingFaceHubDown()


def test_no_internet_connection_is_an_exception():
    """Test that `NoInternetConnection` is an exception."""
    with pytest.raises(NoInternetConnection):
        raise NoInternetConnection()


def test_nan_value_in_model_output():
    """Test that `NaNValueInModelOutput` is an exception."""
    with pytest.raises(NaNValueInModelOutput):
        raise NaNValueInModelOutput()


def test_flash_attention_not_installed():
    """Test that `FlashAttentionNotInstalled` is an exception."""
    with pytest.raises(FlashAttentionNotInstalled):
        raise FlashAttentionNotInstalled()


@pytest.mark.parametrize(
    argnames=["extra"], argvalues=[("test",), ("",)], ids=["extra", "empty_extra"]
)
def test_needs_extra_installed(extra):
    """Test that `NeedsExtraInstalled` is an exception."""
    exception_regex = f".*`{extra}`.*"
    with pytest.raises(NeedsExtraInstalled, match=exception_regex):
        raise NeedsExtraInstalled(extra)


@pytest.mark.parametrize(
    argnames=["cli_argument", "script_argument", "run_with_cli"],
    argvalues=[
        ("--test-argument", "test_argument", True),
        ("--test-argument", "test_argument", False),
        ("--", "", True),
        ("--", "", False),
    ],
    ids=[
        "argument_with_cli",
        "argument_without_cli",
        "empty_argument_with_cli",
        "empty_argument_without_cli",
    ],
)
def test_needs_additional_argument(cli_argument, script_argument, run_with_cli):
    """Test that `NeedsExtraInstalled` is an exception."""
    if run_with_cli:
        exception_regex = f".*`{cli_argument}`.*`scandeval` command.*"
    else:
        exception_regex = f".*`{script_argument}`.*`Benchmarker` class.*"
    with pytest.raises(NeedsAdditionalArgument, match=exception_regex):
        raise NeedsAdditionalArgument(
            cli_argument=cli_argument,
            script_argument=script_argument,
            run_with_cli=run_with_cli,
        )


@pytest.mark.parametrize(argnames=["run_with_cli"], argvalues=[(True,), (False,)])
def test_missing_hugging_face_token(run_with_cli):
    """Test that `NeedsExtraInstalled` is an exception."""
    if run_with_cli:
        exception_regex = ".*`huggingface-cli login`.*"
    else:
        exception_regex = ".*`Benchmarker` class.*"
    with pytest.raises(MissingHuggingFaceToken, match=exception_regex):
        raise MissingHuggingFaceToken(run_with_cli=run_with_cli)
