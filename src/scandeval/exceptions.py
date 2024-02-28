"""Exceptions to used by other functions."""


class InvalidBenchmark(Exception):
    """The (model, dataset) combination cannot be benchmarked."""

    def __init__(
        self, message: str = "This model cannot be benchmarked on the given dataset."
    ):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class InvalidModel(Exception):
    """The model cannot be benchmarked on any datasets."""

    def __init__(
        self, message: str = "The model cannot be benchmarked on any datasets."
    ):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class HuggingFaceHubDown(Exception):
    """The Hugging Face Hub seems to be down."""

    def __init__(self, message: str = "The Hugging Face Hub is currently down."):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class NoInternetConnection(Exception):
    """There seems to be no internet connection."""

    def __init__(self, message: str = "There is currently no internet connection."):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class NaNValueInModelOutput(Exception):
    """There is a NaN value in the model output."""

    def __init__(self, message: str = "There is a NaN value in the model output."):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class FlashAttentionNotInstalled(Exception):
    """The `flash-attn` package has not been installed."""

    def __init__(
        self,
        message: str = (
            "The model you are trying to load requires Flash Attention. To use Flash "
            "Attention, please install the `flash-attn` package, which can be done by "
            "running `pip install -U wheel && FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE "
            "pip install flash-attn --no-build-isolation`."
        ),
    ):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class NeedsExtraInstalled(InvalidModel):
    """The evaluation requires extra to be installed."""

    def __init__(self, extra: str):
        """Initialize the exception.

        Args:
            extra:
                The extra that needs to be installed.
        """
        self.message = (
            f"The model you are trying to load requires the `{extra}` extra to be "
            f"installed. To install the `{extra}` extra, please run `pip install "
            f"scandeval[{extra}]` or `pip install scandeval[all]`."
        )
        super().__init__(self.message)


class NeedsAdditionalArgument(InvalidModel):
    """The evaluation requires additional arguments to the `scandeval` command."""

    def __init__(self, cli_argument: str, script_argument: str, run_with_cli: bool):
        """Initialize the exception.

        Args:
            cli_argument:
                The argument that needs to be passed to the `scandeval` command.
            script_argument:
                The argument that needs to be passed to the `Benchmarker` class.
            run_with_cli:
                Whether the benchmark is being run with the CLI.
        """
        if run_with_cli:
            self.message = (
                f"The model you are trying to load requires the `{cli_argument}` "
                "argument to be passed to the `scandeval` command. Please pass the "
                "argument and try again."
            )
        else:
            self.message = (
                f"The model you are trying to load requires the `{script_argument}` "
                "argument  to be passed to the `Benchmarker` class. Please pass the "
                "argument and try again."
            )
        super().__init__(self.message)


class MissingHuggingFaceToken(InvalidModel):
    """The evaluation requires a Hugging Face token."""

    def __init__(self, run_with_cli: bool):
        """Initialize the exception.

        Args:
            run_with_cli:
                Whether the benchmark is being run with the CLI.
        """
        self.message = (
            "The model you are trying to load requires a Hugging Face token. "
        )
        if run_with_cli:
            self.message += (
                "Please run `huggingface-cli login` to login to the Hugging Face "
                "Hub and try again. Alternatively, you can pass your Hugging Face Hub "
                "token directly to the `scandeval` command with the `--token <token>` "
                "argument."
            )
        else:
            self.message += (
                "Please pass your Hugging Face Hub token to the `Benchmarker` class "
                "with the `token=<token>` argument  to the `Benchmarker` class and try "
                "again. Alternatively, you can also simply pass `token=True` (which is "
                "the default) to the `Benchmarker` class, which assumes that you have "
                "logged into the Hugging Face Hub. This can be done by running "
                "`huggingface-cli login` in the terminal or `from huggingface_hub "
                "import login; login()` in a Python script."
            )
        super().__init__(self.message)
