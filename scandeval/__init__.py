__version__ = '0.0.0'  # noqa

# Block unwanted terminal outputs
from .utils import block_terminal_output; block_terminal_output()

# Set up logging
import logging
format = '%(asctime)s [%(levelname)s] <%(name)s> %(message)s'
logging.basicConfig(level=logging.INFO, format=format)

# Disable parallelisation when tokenizing, as that can lead to errors
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import classes
from .dane import DaneBenchmark  # noqa
from .benchmark import Benchmark  # noqa
