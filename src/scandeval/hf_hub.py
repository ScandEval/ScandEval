"""Functions related to fetching data from the Hugging Face Hub."""

import logging
from collections import defaultdict
from typing import Dict, Optional, Sequence, Union

import yaml
from huggingface_hub import HfApi, ModelFilter
from requests.exceptions import ConnectionError, RequestException

from .config import BenchmarkConfig, ModelConfig
from .datasets import get_config_dir
from .exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


# TODO: Cache this
def get_model_config(model_id: str, benchmark_config: BenchmarkConfig) -> ModelConfig:
    """Fetches configuratino for a model from the Hugging Face Hub.

    Args:
        model_id (str):
            The full Hugging Face Hub ID of the model.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        RuntimeError: If the extracted framework is not recognized.
    """
    # If the model ID specifies a random ID, then return a hardcoded metadata
    # dictionary
    if model_id.startswith("random"):
        model_config = ModelConfig(
            model_id=model_id,
            framework="pytorch",
            task="fill-mask",
            languages=[],
            revision="main",
        )
        return model_config

    # Extract the revision from the model ID, if it is specified
    if "@" in model_id:
        model_id_without_revision, revision = model_id.split("@", 1)
    else:
        model_id_without_revision = model_id
        revision = "main"

    # Extract the author and model name from the model ID
    author: Optional[str]
    if "/" in model_id_without_revision:
        author, model_name = model_id_without_revision.split("/")
    else:
        author = None
        model_name = model_id_without_revision

    # Attempt to fetch model data from the Hugging Face Hub
    try:

        # Define the API object
        api = HfApi()

        # Fetch the model metadata
        models = api.list_models(
            filter=ModelFilter(author=author, model_name=f"{model_name}@{revision}"),
            use_auth_token=benchmark_config.use_auth_token,
        )

        # Check that the model exists. If it does not then raise an error
        if len(models) == 0:
            raise InvalidBenchmark(f"Model {model_id} does not exist.")

        # Fetch the model tags
        tags = models[0].tags

        # Extract the framework, which defaults to PyTorch
        framework = "pytorch"
        if "pytorch" in tags:
            pass
        elif "jax" in tags:
            framework = "jax"
        elif "spacy" in tags:
            framework = "spacy"
        elif "tf" in tags or "tensorflow" in tags or "keras" in tags:
            raise InvalidBenchmark("TensorFlow/Keras models are not supported.")

        # Extract the model task, which defaults to 'fill-mask'
        task = models[0].pipeline_tag
        if task is None or task in ["sentence-similarity", "feature-extraction"]:
            task = "fill-mask"

        # Get list of all language codes
        with (get_config_dir() / "language_codes.yaml").open() as f:
            language_codes = yaml.safe_load(f)

        # Construct the model config
        model_config = ModelConfig(
            model_id=model_id_without_revision,
            framework=framework,
            task=task,
            languages=[tag for tag in tags if tag in language_codes],
            revision=revision,
        )

    # If fetching from the Hugging Face Hub failed then throw a reasonable exception
    except RequestException:
        raise ConnectionError(
            "Connection to the Hugging Face Hub failed. Check your internet "
            "connection and if https://huggingface.co is down."
        )

    # Return the model config
    return model_config


# TODO: Cache this
def get_model_lists(
    languages: Sequence[Optional[str]],
    tasks: Sequence[Optional[str]],
    use_auth_token: bool,
) -> Union[Dict[str, Sequence[str]], None]:
    """Fetches up-to-date model lists.

    Args:
        languages (list of either str or None):
            The language codes of the language to consider. If None is present in the
            list then the models will not be filtered on language.
        tasks (list of either str or None):
            The task to consider. If None is present in the list then the models will
            not be filtered on task.
        use_auth_token (bool):
            Whether to use an authentication token to fetch the model lists.

    Returns:
        dict:
            The keys are filterings of the list, which includes all language codes,
            including 'multilingual', all tasks, as well as 'all'. The values are lists
            of model IDs.
    """
    # Log fetching message
    log_msg = "Fetching list of models"
    if None not in languages:
        log_msg += f" for the languages {languages}"
        if None not in tasks:
            log_msg += f" and tasks {tasks}"
    else:
        if None not in tasks:
            log_msg += f" for the tasks {tasks}"
    log_msg += " from the Hugging Face Hub."
    logger.info(log_msg)

    # Initialise the API
    api = HfApi()

    # Initialise model lists
    model_lists = defaultdict(list)
    for language in languages:
        for task in tasks:

            # Fetch the model list
            models = api.list_models(
                filter=ModelFilter(language=language, task=task),
                use_auth_token=use_auth_token,
            )

            # Extract the model IDs
            model_ids = [model.id for model in models]

            # Store the model IDs
            model_lists["all"].extend(model_ids)
            if language is not None:
                model_lists[language].extend(model_ids)
            if task is not None:
                model_lists[task].extend(model_ids)

    # Add multilingual models manually
    multi_models = [
        "xlm-roberta-large",
        "Peltarion/xlm-roberta-longformer-base-4096",
        "microsoft/xlm-align-base",
        "microsoft/infoxlm-base",
        "microsoft/infoxlm-large",
        "bert-base-multilingual-cased",
        "bert-base-multilingual-uncased",
        "distilbert-base-multilingual-cased",
        "cardiffnlp/twitter-xlm-roberta-base",
    ]
    model_lists["multilingual"] = multi_models
    model_lists["all"].extend(multi_models)

    # Add random models
    random_models = [
        "random-xlmr-base-sequence-clf",
        "random-xlmr-base-token-clf",
        "random-electra-small-sequence-clf",
        "random-electra-small-token-clf",
    ]
    model_lists["all"].extend(random_models)

    # Add some multilingual Danish models manually that have not marked
    # 'da' as their language
    if "da" in languages:
        multi_da_models = [
            "Geotrend/bert-base-en-da-cased",
            "Geotrend/bert-base-25lang-cased",
            "Geotrend/bert-base-en-fr-de-no-da-cased",
            "Geotrend/distilbert-base-en-da-cased",
            "Geotrend/distilbert-base-25lang-cased",
            "Geotrend/distilbert-base-en-fr-de-no-da-cased",
        ]
        model_lists["da"].extend(multi_da_models)
        model_lists["all"].extend(multi_da_models)

    # Add some multilingual Norwegian models manually that have not marked
    # 'no', 'nb' or 'nn' as their language
    if any(lang in languages for lang in ["no", "nb", "nn"]):
        multi_no_models = [
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

    return dict(model_lists)
