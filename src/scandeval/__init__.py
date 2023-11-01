"""
.. include:: ../../README.md
"""

import importlib.metadata
import logging
import os

from dotenv import load_dotenv
from termcolor import colored


# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)


# Loads environment variables
load_dotenv()


# Set up logging
fmt = colored("%(asctime)s", "light_blue") + " ⋅ " + colored("%(message)s", "green")
logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Single GPU setup if we are not in a distributed environment
if os.getenv("WORLD_SIZE") is None:
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"][0]


# Import submodules after setting environment variables since the modules import
# `torch`, which needs to be imported after setting the environment variables
from .benchmarker import Benchmarker  # noqa: E402
from .utils import block_terminal_output  # noqa: E402


# Block unwanted terminal outputs
block_terminal_output()
