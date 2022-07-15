"""Functions related to fetching data from the Hugging Face Hub."""

import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import requests
from huggingface_hub import HfApi, ModelFilter
from requests.exceptions import RequestException

from .config import BenchmarkConfig, Language, ModelConfig
from .exceptions import HuggingFaceHubDown, InvalidBenchmark, NoInternetConnection
from .languages import DA, NB, NN, NO, SV, get_all_languages

logger = logging.getLogger(__name__)


# TODO: Cache this
def get_model_config(model_id: str, benchmark_config: BenchmarkConfig) -> ModelConfig:
    """Fetches configuration for a model from the Hugging Face Hub.

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
            filter=ModelFilter(author=author, model_name=model_name),
            use_auth_token=benchmark_config.use_auth_token,
        )

        # Filter the models to only keep the one with the specified model ID
        models = [
            model for model in models if model.modelId == model_id_without_revision
        ]

        # Check that the model exists. If it does not then raise an error
        if len(models) == 0:
            raise InvalidBenchmark(
                f"The model {model_id} does not exist on the Hugging Face Hub."
            )

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
        model_task = models[0].pipeline_tag
        if model_task is None or model_task in [
            "sentence-similarity",
            "feature-extraction",
        ]:
            model_task = "fill-mask"

        # Get list of all language codes
        language_mapping = get_all_languages()
        language_codes = list(language_mapping.keys())

        # Construct the model config
        model_config = ModelConfig(
            model_id=models[0].modelId,
            framework=framework,
            task=model_task,
            languages=[language_mapping[tag] for tag in tags if tag in language_codes],
            revision=revision,
        )

    # If fetching from the Hugging Face Hub failed then throw a reasonable exception
    except RequestException:

        # Check if it is because the internet is down, by pinging Google
        try:
            requests.get("https://www.google.com")

            # If no errors were raised then Hugging Face Hub is down
            raise HuggingFaceHubDown()

        # Otherwise, if pinging Google also failed, then the internet is down
        except RequestException:
            raise NoInternetConnection()

    # Return the model config
    return model_config


# TODO: Cache this
def get_model_lists(
    languages: Optional[Sequence[Language]],
    tasks: Optional[Sequence[str]],
    use_auth_token: bool,
) -> Dict[str, Sequence[str]]:
    """Fetches up-to-date model lists.

    Args:
        languages (None or sequence of Language objects):
            The language codes of the language to consider. If None then the models
            will not be filtered on language.
        tasks (None or sequence of str):
            The task to consider. If None then the models will not be filtered on task.
        use_auth_token (bool):
            Whether to use an authentication token to fetch the model lists.

    Returns:
        dict:
            The keys are filterings of the list, which includes all language codes,
            including 'multilingual', all tasks, as well as 'all'. The values are lists
            of model IDs.
    """
    # Get list of all languages
    all_languages = list(get_all_languages().values())

    # If no languages are specified, then include all languages
    language_list = all_languages if languages is None else languages

    # Form string of languages
    if len(language_list) == 1:
        language_string = f"the language {language_list[0].name}"
    else:
        language_list = sorted(language_list, key=lambda x: x.name)
        if {lang.code for lang in language_list} == {
            lang.code for lang in all_languages
        }:
            language_string = "all languages"
        else:
            language_string = (
                f"the languages {', '.join(l.name for l in language_list[:-1])} "
                f"and {language_list[-1].name}"
            )

    # Form string of tasks
    if tasks is None:
        task_string = "all model tasks"
    elif len(tasks) == 1:
        task_string = f"the model task {tasks[0]}"
    else:
        tasks = sorted(tasks)
        task_string = f"the model tasks {', '.join(tasks[:-1])} and {tasks[-1]}"

    # Log fetching message
    logger.info(
        f"Fetching list of models for {language_string} and {task_string} from the "
        "Hugging Face Hub."
    )

    # Initialise the API
    api = HfApi()

    # Initialise model lists
    model_lists = defaultdict(list)

    # Do not iterate over all the languages if we are not filtering on language
    language_itr: Sequence[Optional[Language]]
    if {lang.code for lang in language_list} == {lang.code for lang in all_languages}:
        language_itr = [None]
    else:
        language_itr = deepcopy(language_list)

    for language in language_itr:
        for task in tasks or [None]:  # type: ignore

            # Fetch the model list
            models = api.list_models(
                filter=ModelFilter(language=language, task=task),
                use_auth_token=use_auth_token,
            )

            # Filter the models to only keep the ones with the specified language and
            # task
            models = [
                model
                for model in models
                if (language is None or language.code in model.tags)
                and (task is None or model.pipeline_tag == task)
            ]

            # Extract the model IDs
            model_ids = [model.id for model in models]

            # Store the model IDs
            model_lists["all"].extend(model_ids)
            if language is not None:
                model_lists[language.code].extend(model_ids)
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
    model_lists["random"].extend(random_models)
    model_lists["all"].extend(random_models)

    # Add some multilingual Danish models manually that have not marked 'da' as their
    # language
    if DA in language_itr:
        multi_da_models: List[str] = [
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
        multi_sv_models: List[str] = []
        model_lists["sv"].extend(multi_sv_models)
        model_lists["all"].extend(multi_sv_models)

    # Add some multilingual Norwegian models manually that have not marked 'no', 'nb'
    # or 'nn' as their language
    if any(lang in language_itr for lang in [NO, NB, NN]):
        multi_no_models: List[str] = [
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
