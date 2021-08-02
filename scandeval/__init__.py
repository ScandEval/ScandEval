__version__ = '0.0.0'  # noqa

# Block unwanted terminal outputs
from .utils import block_terminal_output; block_terminal_output()

# Set up logging
import logging
from termcolor import colored
format = colored('%(asctime)s [%(levelname)s] <%(name)s>\n', 'green') + \
         colored('    %(message)s', 'yellow')
logging.basicConfig(level=logging.INFO, format=format)

# Disable parallelisation when tokenizing, as that can lead to errors
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import classes
from .dane import DaneBenchmark  # noqa
from .benchmark import Benchmark  # noqa
