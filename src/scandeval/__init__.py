"""Evaluation of language models on mono- or multilingual Scandinavian language tasks."""

import logging
import os
from importlib import metadata

from termcolor import colored

from .benchmarker import Benchmarker  # noqa
from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = metadata.version(__package__)  # noqa


# Block unwanted terminal outputs
block_terminal_output()


# Set up logging
format = colored("%(asctime)s [%(levelname)s] <%(name)s>\n? ", "green") + colored(
    "%(message)s", "yellow"
)
logging.basicConfig(level=logging.INFO, format=format)


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"
