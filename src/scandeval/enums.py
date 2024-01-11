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
        API:
            Accessible via an API.
    """

    PYTORCH = auto()
    JAX = auto()
    API = auto()


class ModelType(AutoStrEnum):
    """The type of a model.

    Attributes:
        FRESH:
            Randomly initialised Hugging Face model.
        HF:
            Model from the Hugging Face Hub.
        LOCAL:
            Locally stored Hugging Face model.
        OPENAI:
            Model from OpenAI.
    """

    FRESH = auto()
    HF = auto()
    LOCAL = auto()
    OPENAI = auto()


class DataType(AutoStrEnum):
    """The data type of the model weights.

    Attributes:
        FP32:
            32-bit floating point.
        FP16:
            16-bit floating point.
        BF16:
            16-bit bfloat.
    """

    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
