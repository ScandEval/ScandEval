"""Utility functions to be used in other scripts."""

import gc
import logging
import os
import random
import re
import warnings

import numpy as np
import pkg_resources
import torch
import transformers.utils.logging as tf_logging
from datasets.utils import disable_progress_bar


def clear_memory():
    """Clears the memory of unused items."""

    # Clear the Python cache
    gc.collect()

    # Empty the CUDA cache
    # TODO: Also empty MPS cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enforce_reproducibility(framework: str, seed: int = 4242):
    """Ensures reproducibility of experiments.

    Args:
        framework (str):
            The framework used for the benchmarking.
        seed (int):
            Seed for the random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if framework in ("pytorch", "jax"):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    return rng


def is_module_installed(module: str) -> bool:
    """Check if a module is installed.

    This is used when dealing with spaCy models, as these are installed as separate
    Python packages.

    Args:
        module (str):
            The name of the module.

    Returns:
        bool:
            Whether the module is installed or not.
    """
    # Get list of all modules, including their versions
    installed_modules_with_versions = list(pkg_resources.working_set)

    # Strip the module versions from the list of modules. Also make the modules lower
    # case and replace dashes with underscores
    installed_modules = [
        re.sub("[0-9. ]", "", str(module)).lower().replace("-", "_")
        for module in installed_modules_with_versions
    ]

    # Check if the module is installed by checking if the module name is in the list
    return module.lower() in installed_modules


def block_terminal_output():
    """Blocks libraries from writing output to the terminal.

    This filters warnings from some libraries, sets the logging level to ERROR for some
    libraries, disabled tokeniser progress bars when using Hugging Face tokenisers, and
    disables most of the logging from the `transformers` library.
    """

    # Ignore miscellaneous warnings
    warnings.filterwarnings(
        "ignore",
        module="torch.nn.parallel*",
        message="Was asked to gather along dimension 0, but all input tensors were "
        "scalars; will instead unsqueeze and return a vector.",
    )
    warnings.filterwarnings("ignore", module="seqeval*")

    # Up the logging level, to disable outputs
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Disable the tokeniser progress bars
    disable_progress_bar()

    # Disable most of the `transformers` logging
    tf_logging.set_verbosity_error()
