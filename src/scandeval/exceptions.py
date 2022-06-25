"""Exceptions to used by other functions."""


class InvalidBenchmark(Exception):
    def __init__(
        self, message: str = "This model cannot be benchmarked on the given dataset."
    ):
        self.message = message
        super().__init__(self.message)
