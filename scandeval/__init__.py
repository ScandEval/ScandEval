from termcolor import colored
import logging
import os
from .benchmark import Benchmark  # noqa
from .datasets import load_dataset  # noqa
from .utils import block_terminal_output

__version__ = '1.2.0'  # noqa

# Block unwanted terminal outputs
block_terminal_output()

# Set up logging
format = colored('%(asctime)s [%(levelname)s] <%(name)s>\n↳ ', 'green') + \
         colored('%(message)s', 'yellow')
logging.basicConfig(level=logging.INFO, format=format)

# Disable parallelisation when tokenizing, as that can lead to errors
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
