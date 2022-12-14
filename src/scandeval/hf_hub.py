"""Functions related to fetching data from the Hugging Face Hub."""

import logging
import re
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Union

from huggingface_hub.hf_api import HfApi, ModelInfo
from huggingface_hub.utils.endpoint_helpers import ModelFilter
from requests.exceptions import RequestException

from .config import BenchmarkConfig, Language, ModelConfig
from .exceptions import HuggingFaceHubDown, InvalidBenchmark, NoInternetConnection
from .languages import DA, NB, NN, NO, SV, get_all_languages
from .utils import internet_connection_available

logger = logging.getLogger(__name__)


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
        RuntimeError:
            If the extracted framework is not recognized.
    """
    # If the model ID specifies a fresh ID, then return a hardcoded metadata
    # dictionary
    if model_id.startswith("fresh"):
        model_config = ModelConfig(
            model_id=model_id,
            framework="pytorch",
            task="fill-mask",
            languages=list(),
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
        api: HfApi = HfApi()

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
        tags: Sequence[str] = models[0].tags

        # Extract the framework, which defaults to PyTorch
        framework = "pytorch"
        if "pytorch" in tags:
            pass
        elif "jax" in tags:
            framework = "jax"
        elif "spacy" in tags:
            raise InvalidBenchmark("SpaCy models are not supported.")
        elif "tf" in tags or "tensorflow" in tags or "keras" in tags:
            raise InvalidBenchmark("TensorFlow/Keras models are not supported.")

        # Extract the model task, which defaults to 'fill-mask'
        model_task: Optional[str] = models[0].pipeline_tag
        if model_task is None:
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
        if internet_connection_available():
            raise HuggingFaceHubDown()
        else:
            raise NoInternetConnection()

    # Return the model config
    return model_config


# TODO: Cache this
def get_model_lists(
    languages: Optional[Sequence[Language]],
    use_auth_token: Union[bool, str],
) -> Dict[str, Sequence[str]]:
    """Fetches up-to-date model lists.

    Args:
        languages (None or sequence of Language objects):
            The language codes of the language to consider. If None then the models
            will not be filtered on language.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.

    Returns:
        dict:
            The keys are filterings of the list, which includes all language codes,
            including 'multilingual', as well as 'all'. The values are lists
            of model IDs.
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
                f"{', '.join(l.name for l in language_list[:-1])} and "
                f"{language_list[-1].name}"
            )

    # Log fetching message
    logger.info(f"Fetching list of {language_string} models from the Hugging Face Hub.")

    # Initialise the API
    api: HfApi = HfApi()

    # Initialise model lists
    model_lists = defaultdict(list)

    # Do not iterate over all the languages if we are not filtering on language
    language_itr: Sequence[Optional[Language]]
    if {lang.code for lang in language_list} == {lang.code for lang in all_languages}:
        language_itr = [None]
    else:
        language_itr = deepcopy(language_list)

    for language in language_itr:

        # Extract the language code
        language_str: Optional[str]
        if language is not None:
            language_str = language.code
        else:
            language_str = None

        # Fetch the model list
        models: List[ModelInfo] = api.list_models(
            filter=ModelFilter(language=language_str),
            use_auth_token=use_auth_token,
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
            in {
                "fill-mask",
                "sentence-similarity",
                "feature-extraction",
                "text-generation",
            }
        ]

        # Extract the model IDs
        model_ids: List[str] = [model.modelId for model in models if model.modelId]

        # Remove models that are too large, and thus needs to be specified manually
        large_regex = re.compile(r"(-|_)x+l(arge)?")
        model_ids = [
            model_id
            for model_id in model_ids
            if re.search(large_regex, model_id) is None
        ]

        # Store the model IDs
        model_lists["all"].extend(model_ids)
        if language is not None:
            model_lists[language.code].extend(model_ids)

    # Add multilingual models manually
    multi_models = [
        "bert-base-multilingual-cased",
        "bert-base-multilingual-uncased",
        "cardiffnlp/twitter-xlm-roberta-base",
        "distilbert-base-multilingual-cased",
        "microsoft/infoxlm-base",
        "microsoft/infoxlm-large",
        "microsoft/mdeberta-v3-base",
        "microsoft/xlm-align-base",
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
        "xlm-roberta-base",
        "xlm-roberta-large",
        "dbmdz/bert-tiny-historic-multilingual-cased",
        "dbmdz/bert-mini-historic-multilingual-cased",
        "dbmdz/bert-base-historic-multilingual-cased",
        "dbmdz/bert-medium-historic-multilingual-cased",
    ]
    model_lists["multilingual"] = multi_models
    model_lists["all"].extend(multi_models)

    # Add fresh models
    fresh_models = ["fresh-xlmr-base", "fresh-electra-small"]
    model_lists["fresh"].extend(fresh_models)
    model_lists["all"].extend(fresh_models)

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

    # Remove banned models
    BANNED_MODELS = [
        r"TransQuest/siamesetransquest-da.*",
        r"M-CLIP/.*",
    ]
    for lang, model_list in model_lists.items():
        model_lists[lang] = [
            model
            for model in model_list
            if not any(re.search(regex, model) is not None for regex in BANNED_MODELS)
        ]

    return dict(model_lists)
