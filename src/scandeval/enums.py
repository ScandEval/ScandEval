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
        HUMAN:
            Human evaluator.
    """

    PYTORCH = auto()
    JAX = auto()
    API = auto()
    HUMAN = auto()


class ModelType(AutoStrEnum):
    """The type of a model.

    Attributes:
        FRESH:
            Randomly initialised Hugging Face model.
        HF_HUB_ENCODER:
            Hugging Face encoder model from the Hub.
        HF_HUB_GENERATIVE:
            Hugging Face generative model from the Hub.
        API:
            Model accessed through an API.
        HUMAN:
            Human evaluator.
    """

    FRESH = auto()
    HF_HUB_ENCODER = auto()
    HF_HUB_GENERATIVE = auto()
    API = auto()
    HUMAN = auto()


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


class BatchingPreference(AutoStrEnum):
    """The preference for batching.

    Attributes:
        NO_PREFERENCE:
            No preference for batching.
        SINGLE_SAMPLE:
            Single sample batching.
        ALL_AT_ONCE:
            All samples at once batching.
    """

    NO_PREFERENCE = auto()
    SINGLE_SAMPLE = auto()
    ALL_AT_ONCE = auto()


class TaskGroup(AutoStrEnum):
    """The overall task group of a task.

    Attributes:
        SEQUENCE_CLASSIFICATION:
            Classification of documents.
        MULTIPLE_CHOICE_CLASSIFICATION:
            Classification of documents with multiple-choice options.
        TOKEN_CLASSIFICATION:
            Token-level classification.
        QUESTION_ANSWERING:
            Extractive question answering.
        TEXT_TO_TEXT:
            Text-to-text generation.
        SPEED:
            Speed benchmark.
    """

    SEQUENCE_CLASSIFICATION = auto()
    MULTIPLE_CHOICE_CLASSIFICATION = auto()
    TOKEN_CLASSIFICATION = auto()
    QUESTION_ANSWERING = auto()
    TEXT_TO_TEXT = auto()
    SPEED = auto()
