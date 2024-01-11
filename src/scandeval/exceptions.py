"""Exceptions to used by other functions."""


class InvalidBenchmark(Exception):
    """The benchmark is invalid and cannot be run."""

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
            "The model you are trying to load requires Flash "
            "Attention. To use Flash Attention, please install "
            "the `flash-attn` package, which can be done by "
            "running `pip install -U wheel && "
            "FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install "
            "flash-attn --no-build-isolation`."
        ),
    ):
        """Initialize the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)
