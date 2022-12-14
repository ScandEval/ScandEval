"""Utility functions to be used in other scripts."""

import gc
import importlib
import logging
import os
import random
import re
import warnings
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import pkg_resources
import requests
import torch
from datasets.utils import disable_progress_bar
from requests.exceptions import RequestException
from transformers import logging as tf_logging
from transformers.tokenization_utils import PreTrainedTokenizer

from .exceptions import InvalidBenchmark


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
        torch.use_deterministic_algorithms(True, warn_only=True)
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
    tf_logging._default_log_level = logging.CRITICAL
    tf_logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)


def get_class_by_name(
    class_name: Union[str, Sequence[str]],
    module_name: Optional[str] = None,
) -> Union[None, Type]:
    """Get a class by its name.

    Args:
        class_name (str or list of str):
            The name of the class, written in kebab-case. The corresponding class name
            must be the same, but written in PascalCase, and lying in a module with the
            same name, but written in snake_case. If a list of strings is passed, the
            first class that is found is returned.
        module_name (str, optional):
            The name of the module where the class is located. If None then the module
            name is assumed to be the same as the class name, but written in
            snake_case. Defaults to None.

    Returns:
        type or None:
            The class. If the class is not found, None is returned.
    """
    # Ensure that `class_name` is a sequence
    if isinstance(class_name, str):
        class_name = [class_name]

    # Loop over the class names
    for name in class_name:

        # Get the snake_case and PascalCase version of the class name
        name_snake = name.replace("-", "_")
        name_pascal = kebab_to_pascal(name)

        # Import the module
        try:
            if not module_name:
                module_name = f"scandeval.{name_snake}"
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            module_name = None
            continue

        # Get the class from the module
        try:
            class_: Type = getattr(module, name_pascal)
        except AttributeError:
            module_name = None
            continue

        # Return the class
        return class_

    # If the class could not be found, return None
    return None


def kebab_to_pascal(kebab_string: str) -> str:
    """Converts a kebab-case string to PascalCase.

    Args:
        kebab_string (str):
            The kebab-case string.

    Returns:
        str:
            The PascalCase string.
    """
    return "".join(word.title() for word in kebab_string.split("-"))


def handle_error(
    e: Exception, per_device_train_batch_size: int, gradient_accumulation_steps: int
) -> Tuple[int, int]:
    """Handle an error that occurred during the benchmarking process.

    Args:
        e (Exception):
            The exception that was raised.
        per_device_train_batch_size (int):
            The batch size used for training.
        gradient_accumulation_steps (int):
            The number of gradient accumulation steps.

    Returns:
        pair of int:
            The batch size and gradient accumulation steps to use.
    """
    # We assume that all these CUDA errors are caused by
    # insufficient GPU memory
    # TODO: Handle MPS out of memory as well
    cuda_errs = ["CUDA out of memory", "CUDA error"]

    # If it is an unknown error, then simply report it
    if all([err not in str(e) for err in cuda_errs]):
        raise InvalidBenchmark(str(e))

    # If it is a CUDA memory error, then reduce batch size and up gradient
    # accumulation
    if per_device_train_batch_size == 1:
        raise InvalidBenchmark("CUDA out of memory, even with a batch size of 1!")
    return per_device_train_batch_size // 2, gradient_accumulation_steps * 2


def internet_connection_available() -> bool:
    """Checks if internet connection is available by pinging google.com.

    Returns:
            bool:
                Whether or not internet connection is available.
    """
    try:
        requests.get("https://www.google.com")
        return True
    except RequestException:
        return False


def get_special_token_metadata(tokenizer: PreTrainedTokenizer) -> dict:
    """Get the special token metadata for a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer):
            The tokenizer.

    Returns:
        dict:
            The special token metadata.
    """
    # Create some test input IDs, to check if the tokenizer is adding special tokens
    test_input_ids = tokenizer("Test").input_ids

    # Extract the CLS token IDs from the tokenizer, if it's using them
    has_cls_token = True
    if tokenizer.cls_token_id in test_input_ids:
        cls_token_id = tokenizer.cls_token_id
        cls_token = tokenizer.cls_token
    elif tokenizer.bos_token_id in test_input_ids:
        cls_token_id = tokenizer.bos_token_id
        cls_token = tokenizer.bos_token
    elif tokenizer.cls_token is not None:
        cls_token_id = tokenizer.cls_token_id
        cls_token = tokenizer.cls_token
        has_cls_token = False
    else:
        cls_token_id = tokenizer.bos_token_id
        cls_token = tokenizer.bos_token
        has_cls_token = False

    # Extract the SEP token IDs from the tokenizer, if it's using them
    has_sep_token = True
    if tokenizer.sep_token_id in test_input_ids:
        sep_token = tokenizer.sep_token
    elif tokenizer.eos_token_id in test_input_ids:
        sep_token = tokenizer.eos_token
    elif tokenizer.sep_token is not None:
        sep_token = tokenizer.sep_token
        has_sep_token = False
    else:
        sep_token = tokenizer.eos_token
        has_sep_token = False

    return dict(
        cls_token_id=cls_token_id,
        cls_token=cls_token,
        sep_token=sep_token,
        has_cls_token=has_cls_token,
        has_sep_token=has_sep_token,
    )
