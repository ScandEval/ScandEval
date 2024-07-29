"""Utility functions to be used in other scripts."""

import gc
import importlib
import importlib.util
import logging
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Type

import numpy as np
import pkg_resources
import requests
import torch
from datasets.utils import disable_progress_bar
from huggingface_hub import HfApi, ModelFilter
from jinja2 import TemplateError
from requests.exceptions import RequestException
from transformers import GenerationConfig
from transformers import logging as tf_logging

from .enums import Framework
from .exceptions import NaNValueInModelOutput
from .languages import DA, NB, NN, NO, SV, get_all_languages
from .openai_models import OpenAITokenizer

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo
    from transformers import PreTrainedModel

    from .config import Language
    from .protocols import GenerativeModel, Tokenizer
    from .types import Predictions

logger = logging.getLogger(__package__)


if importlib.util.find_spec("ray") is not None:
    import ray


# This is used as input to generative models; it cannot be a special token
DUMMY_FILL_VALUE = 100


GENERATIVE_MODEL_TASKS = [
    "text-generation",
    "conversational",
    # "text2text-generation",  # TODO: Add this when we support it
]


GENERATIVE_DATASET_TASKS = ["knowledge", "common-sense-reasoning"]


GENERATIVE_DATASET_SUPERTASKS = ["text-to-text", "text-modelling"]


SUPERTASKS_USING_LOGPROBS = ["sequence-classification"]


METRIC_ATTRIBUTES_TAKING_UP_MEMORY = ["cached_bertscorer"]


def create_model_cache_dir(cache_dir: str, model_id: str) -> str:
    """Create cache directory for a model.

    Args:
        cache_dir:
            The cache directory.
        model_id:
            The model ID.

    Returns:
        The path to the cache directory.
    """
    # to avoid nesting due to models name containing '/'
    _model_id = model_id.replace("/", "--")
    cache_dir_path = Path(cache_dir) / "model_cache" / _model_id
    return str(cache_dir_path)


def clear_memory():
    """Clears the memory of unused items."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def enforce_reproducibility(framework: Framework | str, seed: int = 4242):
    """Ensures reproducibility of experiments.

    Args:
        framework:
            The framework used for the benchmarking.
        seed:
            Seed for the random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if framework in (Framework.PYTORCH, Framework.JAX):
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
        module:
            The name of the module.

    Returns:
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
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        module="torch.nn.parallel*",
        message="Was asked to gather along dimension 0, but all input tensors were "
        "scalars; will instead unsqueeze and return a vector.",
    )
    warnings.filterwarnings("ignore", module="seqeval*")

    # Up the logging level, to disable outputs
    logging.getLogger("filelock").setLevel(logging.CRITICAL)
    logging.getLogger("absl").setLevel(logging.CRITICAL)
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.CRITICAL)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.CRITICAL)
    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.engine.llm_engine").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.transformers_utils.tokenizer").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.core.scheduler").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.model_executor.weight_utils").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.worker").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.CRITICAL)

    # This suppresses vLLM logging
    os.environ["LOG_LEVEL"] = "CRITICAL"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    if importlib.util.find_spec("ray") is not None:
        ray._private.worker._worker_logs_enabled = False

    # Disable the tokeniser progress bars
    disable_progress_bar()

    # Disable most of the `transformers` logging
    tf_logging._default_log_level = logging.CRITICAL
    tf_logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)


def get_class_by_name(
    class_name: str | list[str], module_name: str | None = None
) -> Type | None:
    """Get a class by its name.

    Args:
        class_name:
            The name of the class, written in kebab-case. The corresponding class name
            must be the same, but written in PascalCase, and lying in a module with the
            same name, but written in snake_case. If a list of strings is passed, the
            first class that is found is returned.
        module_name:
            The name of the module where the class is located. If None then the module
            name is assumed to be the same as the class name, but written in
            snake_case. Defaults to None.

    Returns:
        The class. If the class is not found, None is returned.
    """
    # Ensure that `class_name` is a sequence
    if isinstance(class_name, str):
        class_name = [class_name]

    # Loop over the class names
    error_messages = list()
    for name in class_name:
        # Get the snake_case and PascalCase version of the class name
        name_snake = name.replace("-", "_")
        name_pascal = kebab_to_pascal(kebab_string=name)

        # Import the module
        try:
            if not module_name:
                module_name = f"scandeval.{name_snake}"
            module = importlib.import_module(name=module_name)
        except ModuleNotFoundError as e:
            error_messages.append(str(e))
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

    if error_messages:
        errors = "\n- " + "\n- ".join(error_messages)
        logger.debug(
            f"Could not find the class with the name(s) {', '.join(class_name)}. The "
            f"following error messages were raised: {errors}"
        )

    # If the class could not be found, return None
    return None


def kebab_to_pascal(kebab_string: str) -> str:
    """Converts a kebab-case string to PascalCase.

    Args:
        kebab_string:
            The kebab-case string.

    Returns:
        The PascalCase string.
    """
    return "".join(word.title() for word in kebab_string.split("-"))


def internet_connection_available() -> bool:
    """Checks if internet connection is available by pinging google.com.

    Returns:
        Whether or not internet connection is available.
    """
    try:
        requests.get("https://www.google.com")
        return True
    except RequestException:
        return False


def get_special_token_metadata(tokenizer: "Tokenizer") -> dict:
    """Get the special token metadata for a tokenizer.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
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


def get_huggingface_model_lists(
    languages: list["Language"] | None, token: bool | str | None
) -> dict[str, list[str]]:
    """Fetches up-to-date model lists from the Hugging Face Hub.

    Args:
        languages:
            The language codes of the language to consider. If None then the models
            will not be filtered on language.
        token:
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.

    Returns:
        The keys are filterings of the list, which includes all language codes,
        including 'multilingual', as well as 'all'. The values are lists of model IDs.
    """
    # Get list of all languages
    all_languages = list(get_all_languages().values())

    # If no languages are specified, then include all languages
    language_list = all_languages if languages is None else languages

    # Form string of languages
    if len(language_list) == 1:
        language_string = language_list[0].name
    else:
        language_list = sorted(language_list, key=lambda x: x.name)
        if {lang.code for lang in language_list} == {
            lang.code for lang in all_languages
        }:
            language_string = "all"
        else:
            # Remove generic 'Norwegian' from the list of languages if both 'Bokmål'
            # and 'Nynorsk' already exist in the list
            if all([lang in language_list for lang in [NO, NB, NN]]):
                language_list = [lang for lang in language_list if lang != NO]

            language_string = (
                f"{', '.join(lang.name for lang in language_list[:-1])} and "
                f"{language_list[-1].name}"
            )

    # Log fetching message
    logger.info(f"Fetching list of {language_string} models from the Hugging Face Hub.")

    # Initialise the API
    api: HfApi = HfApi()

    # Initialise model lists
    model_lists = defaultdict(list)

    # Do not iterate over all the languages if we are not filtering on language
    language_itr: list["Language | None"]
    if {lang.code for lang in language_list} == {lang.code for lang in all_languages}:
        language_itr = [None]
    else:
        language_itr = deepcopy(language_list)  # type: ignore[arg-type]

    for language in language_itr:
        # Extract the language code
        language_str: str | None
        if language is not None:
            language_str = language.code
        else:
            language_str = None

        # Fetch the model list
        models: list["ModelInfo"] = list(
            api.list_models(filter=ModelFilter(language=language_str), token=token)
        )

        # Filter the models to only keep the ones with the specified language
        models = [
            model
            for model in models
            if (language is None or language.code in model.tags)
        ]

        # Only keep the models which are not finetuned
        models = [
            model
            for model in models
            if model.pipeline_tag is None
            or model.pipeline_tag
            in {"fill-mask", "sentence-similarity", "feature-extraction"}
            | set(GENERATIVE_MODEL_TASKS)
        ]

        # Extract the model IDs
        model_ids: list[str] = [model.modelId for model in models if model.modelId]

        # Remove models that have "finetuned" in their name
        model_ids = [
            model_id for model_id in model_ids if "finetuned" not in model_id.lower()
        ]

        # Store the model IDs
        model_lists["all"].extend(model_ids)
        if language is not None:
            model_lists[language.code].extend(model_ids)

    # Add multilingual models manually
    multi_models = [
        "google-bert/bert-base-multilingual-cased",
        "google-bert/bert-base-multilingual-uncased",
        "distilbert-base-multilingual-cased",
        "distilbert/cardiffnlp/twitter-xlm-roberta-base",
        "microsoft/infoxlm-base",
        "microsoft/infoxlm-large",
        "microsoft/xlm-align-base",
        "microsoft/mdeberta-v3-base",
        "setu4993/LaBSE",
        "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",
        "sentence-transformers/distiluse-base-multilingual-cased",
        "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "sentence-transformers/quora-distilbert-multilingual",
        "sentence-transformers/stsb-xlm-r-multilingual",
        "sentence-transformers/use-cmlm-multilingual",
        "studio-ousia/mluke-base",
        "studio-ousia/mluke-large",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "dbmdz/bert-tiny-historic-multilingual-cased",
        "dbmdz/bert-mini-historic-multilingual-cased",
        "dbmdz/bert-base-historic-multilingual-cased",
        "dbmdz/bert-medium-historic-multilingual-cased",
    ]
    model_lists["multilingual"] = multi_models
    model_lists["all"].extend(multi_models)

    # Add fresh models
    fresh_models = ["fresh-xlm-roberta-base", "fresh-electra-small"]
    model_lists["fresh"].extend(fresh_models)
    model_lists["all"].extend(fresh_models)

    # Add some multilingual Danish models manually that have not marked 'da' as their
    # language
    if DA in language_itr:
        multi_da_models: list[str] = [
            "Geotrend/bert-base-en-da-cased",
            "Geotrend/bert-base-25lang-cased",
            "Geotrend/bert-base-en-fr-de-no-da-cased",
            "Geotrend/distilbert-base-en-da-cased",
            "Geotrend/distilbert-base-25lang-cased",
            "Geotrend/distilbert-base-en-fr-de-no-da-cased",
        ]
        model_lists["da"].extend(multi_da_models)
        model_lists["all"].extend(multi_da_models)

    # Add some multilingual Swedish models manually that have not marked 'sv' as their
    # language
    if SV in language_itr:
        multi_sv_models: list[str] = []
        model_lists["sv"].extend(multi_sv_models)
        model_lists["all"].extend(multi_sv_models)

    # Add some multilingual Norwegian models manually that have not marked 'no', 'nb'
    # or 'nn' as their language
    if any(lang in language_itr for lang in [NO, NB, NN]):
        multi_no_models: list[str] = [
            "Geotrend/bert-base-en-no-cased",
            "Geotrend/bert-base-25lang-cased",
            "Geotrend/bert-base-en-fr-de-no-da-cased",
            "Geotrend/distilbert-base-en-no-cased",
            "Geotrend/distilbert-base-25lang-cased",
            "Geotrend/distilbert-base-en-fr-de-no-da-cased",
        ]
        model_lists["no"].extend(multi_no_models)
        model_lists["all"].extend(multi_no_models)

    # Remove duplicates from the lists
    for lang, model_list in model_lists.items():
        model_lists[lang] = list(set(model_list))

    # Remove banned models
    BANNED_MODELS = [
        r"TransQuest/siamesetransquest-da.*",
        r"M-CLIP/.*",
        r".*/.*CTRL.*",  # TEMP
    ]
    for lang, model_list in model_lists.items():
        model_lists[lang] = [
            model
            for model in model_list
            if not any(re.search(regex, model) is not None for regex in BANNED_MODELS)
        ]

    return dict(model_lists)


class HiddenPrints:
    """Context manager which removes all terminal output."""

    def __enter__(self):
        """Enter the context manager."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def model_is_generative(model: "PreTrainedModel | GenerativeModel") -> bool:
    """Check if a model is generative or not.

    Args:
        model:
            The model to check.

    Returns:
        Whether the model is generative or not.
    """
    known_generative_models = ["VLLMModel", "OpenAIModel"]
    if any(model.__class__.__name__ == name for name in known_generative_models):
        return True

    try:
        dummy_inputs = torch.tensor(
            [[DUMMY_FILL_VALUE]], device=model.device, dtype=torch.long
        )
        generation_config = GenerationConfig(
            max_new_tokens=1,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
        )
        model.generate(inputs=dummy_inputs, generation_config=generation_config)
        return True
    except (NotImplementedError, TypeError) as e:
        logger.debug(
            f"The model was found not to be generative, as an error occurred: {e}"
        )
        return False


def raise_if_model_output_contains_nan_values(model_output: "Predictions") -> None:
    """Raise an exception if the model output contains NaN values.

    Args:
        model_output:
            The model output to check.

    Raises:
        If the model output contains NaN values.
    """
    if isinstance(model_output, np.ndarray):
        if model_output.dtype == np.float32 and np.isnan(model_output).any():
            raise NaNValueInModelOutput()
    elif len(model_output) > 0:
        if isinstance(model_output[0], str):
            if any(x != x for x in model_output):
                raise NaNValueInModelOutput()
        elif len(model_output[0]) > 0:
            if any(x != x for sublist in model_output for x in sublist):
                raise NaNValueInModelOutput()


def should_prompts_be_stripped(
    labels_to_be_generated: list[str], tokenizer: "Tokenizer"
) -> bool:
    """Determine if we should strip the prompts for few-shot evaluation.

    This is the case if the tokenizer needs to include the space as part of the label
    token. The strategy is thus to tokenize a label with a preceeding colon (as in the
    prompts), i.e., ": positive", and check if the tokenization starts with the tokens
    of ": ". If this is the case, then we should not strip the prompts, since the
    tokenizer produces the whitespace token separately.

    Args:
        labels_to_be_generated:
            The labels that are to be generated.
        tokenizer:
            The tokenizer used to tokenize the labels.

    Returns:
        Whether we should strip the prompts.
    """
    strip_prompts = True
    for label in labels_to_be_generated:
        colon_tokens = tokenizer(": ", add_special_tokens=False).input_ids
        label_tokens = tokenizer(": " + label, add_special_tokens=False).input_ids

        if isinstance(colon_tokens, torch.Tensor):
            colon_tokens = list(colon_tokens.squeeze(0))
        if isinstance(label_tokens, torch.Tensor):
            label_tokens = list(label_tokens.squeeze(0))

        label_tokens_start_with_colon_tokens = (
            label_tokens[: len(colon_tokens)] == colon_tokens
        )
        if label_tokens_start_with_colon_tokens:
            strip_prompts = False

    return strip_prompts


def should_prefix_space_be_added_to_labels(
    labels_to_be_generated: list[str], tokenizer: "Tokenizer"
) -> bool:
    """Determine if we should add a prefix space to the labels.

    This is the case if the prompts are stripped and the tokenizer doesn't
    automatically add prefix whitespaces to the labels.

    Args:
        labels_to_be_generated:
            The labels that are to be generated.
        tokenizer:
            The tokenizer used to tokenize the labels.

    Returns:
        Whether we should add a prefix space to the labels.
    """
    # We don't add a prefix space to OpenAI models, since they output strings directly,
    # and we always strip these for token ID consistency
    if isinstance(tokenizer, OpenAITokenizer):
        return False

    if not should_prompts_be_stripped(
        labels_to_be_generated=labels_to_be_generated, tokenizer=tokenizer
    ):
        return False

    whitespace_token = tokenizer.convert_ids_to_tokens(
        ids=tokenizer(" ", add_special_tokens=False).input_ids[0]
    )[0]

    add_prefix_space = True
    for label in labels_to_be_generated:
        label_tokens = tokenizer(label, add_special_tokens=False).input_ids
        if isinstance(label_tokens, torch.Tensor):
            label_tokens = list(label_tokens.squeeze(0))
        first_label_token: int = int(label_tokens[0])
        first_character_of_label = tokenizer.convert_ids_to_tokens(first_label_token)[0]
        has_prefix_space = first_character_of_label == whitespace_token
        if has_prefix_space:
            add_prefix_space = False
            break

    return add_prefix_space


def get_end_of_chat_token_ids(tokenizer: "Tokenizer") -> list[int] | None:
    """Get the end token ID for chat models.

    This is only relevant for tokenizers with a chat template.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
        The token IDs used to end chats, or None if the tokenizer does not have a chat
        template.

    Raises:
        ValueError:
            If the end-of-chat token could not be located.
    """
    if tokenizer.chat_template is None:
        return None

    token_ids = tokenizer.apply_chat_template(
        conversation=[dict(role="user", content="X")]
    )
    assert isinstance(token_ids, list)

    for idx, token in enumerate(tokenizer.convert_ids_to_tokens(token_ids)):
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert isinstance(token_id, int)
        token = tokenizer.decode([token_id])
        if "X" in token:
            x_token_index = idx
            break
    else:
        raise ValueError("Could not locate the end-of-chat token for the model.")

    end_of_chat_tokens = token_ids[x_token_index + 1 :]
    if len(end_of_chat_tokens) == 0:
        return None
    return end_of_chat_tokens


def convert_prompt_to_instruction(prompt: str, tokenizer: "Tokenizer") -> str:
    """Convert a prompt to an instruction.

    Note that it is expected that the prompt has the following format:

    ```
    <prefix prompt>

    [<example prefix>: <example>
    <label prefix>: <label>

    <example prefix>: <example>
    <label prefix>: <label>

    (...)

    <example prefix>: <example>
    <label prefix>: <label>]

    <example prefix>: <example>
    <label prefix>:
    ```

    Here the part in square brackets is optional, containing few-shot examples.

    Args:
        prompt:
            The prompt.
        tokenizer:
            The tokenizer.

    Returns:
        The instruction.
    """
    if tokenizer.chat_template is None:
        return prompt

    chat_template_kwargs = dict(
        chat_template=tokenizer.chat_template,
        add_generation_prompt=True,
        tokenize=False,
    )

    prompt_has_prefix = (
        len(prompt.split("\n\n")) > 1 and len(prompt.split("\n\n")[0].split("\n")) == 1
    )
    if not prompt_has_prefix:
        raise ValueError(f"The prompt doesn't have a prefix: {prompt!r}")

    # Split up the prompt into its main components
    prompt_prefix = prompt.split("\n\n")[0]
    main_prompt = "\n".join(prompt.split("\n")[2:-1])
    label_prefix = prompt.split("\n")[-1]

    try:
        instruction_prompt = tokenizer.apply_chat_template(
            conversation=[
                dict(role="system", content=prompt_prefix),
                dict(role="user", content=main_prompt),
            ],
            **chat_template_kwargs,
        )
    except TemplateError:
        instruction_prompt = tokenizer.apply_chat_template(
            conversation=[
                dict(role="user", content=prompt_prefix + "\n\n" + main_prompt)
            ],
            **chat_template_kwargs,
        )
    assert isinstance(instruction_prompt, str)

    instruction_prompt += label_prefix

    return instruction_prompt


def scramble(text: str) -> str:
    """Scramble a string in a bijective manner.

    Args:
        text:
            The string to scramble.

    Returns:
        The scrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(text))
    scrambled = "".join(text[i] for i in permutation)
    return scrambled


def unscramble(scrambled_text: str) -> str:
    """Unscramble a string in a bijective manner.

    Args:
        scrambled_text:
            The scrambled string to unscramble.

    Returns:
        The unscrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(scrambled_text))
    inverse_permutation = np.argsort(permutation)
    unscrambled = "".join(scrambled_text[i] for i in inverse_permutation)
    return unscrambled
