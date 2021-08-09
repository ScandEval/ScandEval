from termcolor import colored
import logging
import os
from .benchmark import Benchmark  # noqa
from .angry_tweets import AngryTweetsBenchmark  # noqa
from .dane import DaneBenchmark  # noqa
from .ddt_pos import DdtPosBenchmark  # noqa
from .dkhate import DkHateBenchmark  # noqa
from .europarl1 import Europarl1Benchmark  # noqa
from .europarl2 import Europarl2Benchmark  # noqa
from .lcc1 import Lcc1Benchmark  # noqa
from .lcc2 import Lcc2Benchmark  # noqa
from .twitter_sent import TwitterSentBenchmark  # noqa
from .utils import block_terminal_output

__version__ = '0.2.0'  # noqa

# Block unwanted terminal outputs
block_terminal_output()

# Set up logging
format = colored('%(asctime)s [%(levelname)s] <%(name)s>\n↳ ', 'green') + \
         colored('%(message)s', 'yellow')
logging.basicConfig(level=logging.INFO, format=format)

# Disable parallelisation when tokenizing, as that can lead to errors
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
