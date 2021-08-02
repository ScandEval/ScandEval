__version__ = '0.0.0'  # noqa

from .utils import block_terminal_output; block_terminal_output()

import logging
format = '%(asctime)s [%(levelname)s] <%(name)s> %(message)s'
logging.basicConfig(level=logging.INFO, format=format)

from .dane import DaneBenchmark  # noqa
from .benchmark import Benchmark  # noqa
