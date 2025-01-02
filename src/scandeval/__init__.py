"""ScandEval - A benchmarking framework for language models."""

import importlib.metadata
import os

from dotenv import load_dotenv

from .utils import block_terminal_output

# Block unwanted terminal outputs
block_terminal_output()
breakpoint()


from .benchmarker import Benchmarker  # noqa: E402

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version("scandeval")


# Loads environment variables
load_dotenv()


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Set amount of threads per GPU - this is the default and is only set to prevent a
# warning from showing
os.environ["OMP_NUM_THREADS"] = "1"


# Set the HF_TOKEN env var to copy the HUGGINGFACE_API_KEY env var, as vLLM uses the
# former and LiteLLM uses the latter
if os.getenv("HUGGINGFACE_API_KEY"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]
