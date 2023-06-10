"""The different types of models that can be benchmarked."""

from typing import Type

from .base import ModelSetup
from .fresh import FreshModelSetup
from .hf import HFModelSetup
from .local import LocalModelSetup
from .openai import OpenAIModelSetup

MODEL_SETUP_CLASSES: list[Type[ModelSetup]] = [
    FreshModelSetup,
    LocalModelSetup,
    HFModelSetup,
    OpenAIModelSetup,
]
