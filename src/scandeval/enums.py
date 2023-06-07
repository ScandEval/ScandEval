"""Enums used in the project."""

from enum import Enum, auto


class AutoStrEnum(str, Enum):
    """StrEnum where auto() returns the field name in lower case."""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list
    ) -> str:
        return name.lower()


class Device(AutoStrEnum):
    """The compute device to use for the evaluation.

    Attributes:
        CPU:
            CPU device.
        MPS:
            MPS GPU, used in M-series MacBooks.
        CUDA:
            CUDA GPU, used with NVIDIA GPUs.
    """

    CPU = auto()
    MPS = auto()
    CUDA = auto()


class Framework(AutoStrEnum):
    """The framework of a model.

    Attributes:
        PYTORCH:
            PyTorch framework.
        JAX:
            JAX framework.
        SPACY:
            spaCy framework.
        OPENAI:
            OpenAI framework.
    """

    PYTORCH = auto()
    JAX = auto()
    SPACY = auto()
    OPENAI = auto()
