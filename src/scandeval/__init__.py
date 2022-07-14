"""
.. include:: ../../README.md
"""

import logging
import os

import pkg_resources
from termcolor import colored

from .benchmarker import Benchmarker  # noqa
from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("scandeval").version


# Block unwanted terminal outputs
block_terminal_output()


# Set up logging
fmt = colored("%(asctime)s [%(levelname)s] <%(name)s>\n↳ ", "green") + colored(
    "%(message)s", "yellow"
)
logging.basicConfig(level=logging.INFO, format=fmt)


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"
