"""The different types of models that can be benchmarked."""

from typing import Type

from ..protocols import ModelSetup
from .fresh import FreshModelSetup
from .hf import HFModelSetup
from .local import LocalModelSetup
from .openai import OpenAIModelSetup

# Note that the order of the model setup classes is important, as the first model setup
# that can load a model will be used
MODEL_SETUP_CLASSES: list[Type[ModelSetup]] = [
    FreshModelSetup,
    LocalModelSetup,
    HFModelSetup,
    OpenAIModelSetup,
]
