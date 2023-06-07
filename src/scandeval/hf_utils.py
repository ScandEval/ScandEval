"""Utility functions related to Hugging Face models."""

from huggingface_hub.hf_api import HfApi, ModelInfo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.utils.endpoint_helpers import ModelFilter
from requests.exceptions import RequestException

from .config import BenchmarkConfig, ModelConfig
from .enums import Framework
from .exceptions import HuggingFaceHubDown, InvalidBenchmark, NoInternetConnection
from .languages import get_all_languages
from .utils import internet_connection_available


def get_hf_model_config(
    model_id: str, benchmark_config: BenchmarkConfig
) -> ModelConfig:
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
            framework=Framework.PYTORCH,
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
    author: str | None
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
        tags: list[str] = models[0].tags

        # Extract the framework, which defaults to PyTorch
        framework = Framework.PYTORCH
        if "pytorch" in tags:
            pass
        elif "jax" in tags:
            framework = Framework.JAX
        elif "spacy" in tags:
            raise InvalidBenchmark("SpaCy models are not supported.")
        elif "tf" in tags or "tensorflow" in tags or "keras" in tags:
            raise InvalidBenchmark("TensorFlow/Keras models are not supported.")

        # Extract the model task, which defaults to 'fill-mask'
        model_task: str | None = models[0].pipeline_tag
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

    return model_config


def get_hf_hub_model_info(model_id: str, use_auth_token: bool | str) -> ModelInfo:
    """Fetches information about a model on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.

    Returns:
        ModelInfo:
            The model information.

    Raises:
        RepositoryNotFoundError:
            If the model does not exist on the Hugging Face Hub.
        HuggingFaceHubDown:
            If the model id exists, we are able to request other adresses,
            but we failed to fetch the desired model.
        NoInternetConnection:
            We are not able to request other adresses.
    """
    # Extract the revision from the model_id, if present
    model_id, revision = model_id.split("@") if "@" in model_id else (model_id, "main")

    # Connect to the Hugging Face Hub API
    hf_api = HfApi()

    # Get the model info, and return it
    try:
        token = None if isinstance(use_auth_token, bool) else use_auth_token
        return hf_api.model_info(
            repo_id=model_id,
            revision=revision,
            token=token,
        )

    # If the repository was not found on Hugging Face Hub then raise that error
    except RepositoryNotFoundError as e:
        raise e

    # If fetching from the Hugging Face Hub failed in a different way then throw a
    # reasonable exception
    except RequestException:
        if internet_connection_available():
            raise HuggingFaceHubDown()
        else:
            raise NoInternetConnection()


def model_exists_on_hf_hub(
    model_id: str,
    use_auth_token: bool | str,
) -> bool | None:
    """Checks whether a model exists on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.

    Returns:
        bool:
            If model exists on the Hugging Face Hub or not.
    """
    try:
        get_hf_hub_model_info(model_id=model_id, use_auth_token=use_auth_token)
        return True
    except RepositoryNotFoundError:
        return False
