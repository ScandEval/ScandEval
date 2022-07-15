"""Exceptions to used by other functions."""


class InvalidBenchmark(Exception):
    def __init__(
        self, message: str = "This model cannot be benchmarked on the given dataset."
    ):
        self.message = message
        super().__init__(self.message)


class HuggingFaceHubDown(Exception):
    def __init__(self, message: str = "The Hugging Face Hub is currently down."):
        self.message = message
        super().__init__(self.message)


class NoInternetConnection(Exception):
    def __init__(self, message: str = "There is currently no internet connection."):
        self.message = message
        super().__init__(self.message)
