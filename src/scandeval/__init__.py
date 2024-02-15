"""ScandEval - A benchmarking framework for language models."""

import importlib.metadata
import logging
import os
import sys

from dotenv import load_dotenv
from termcolor import colored

from .benchmarker import Benchmarker
from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)


# Block unwanted terminal outputs
block_terminal_output()


# Loads environment variables
load_dotenv()


# Set up logging
fmt = colored("%(asctime)s", "light_blue") + " ⋅ " + colored("%(message)s", "green")
logging.basicConfig(
    level=logging.CRITICAL if hasattr(sys, "_called_from_test") else logging.INFO,
    format=fmt,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Set amount of threads per GPU - this is the default and is only set to prevent a
# warning from showing
os.environ["OMP_NUM_THREADS"] = "1"
