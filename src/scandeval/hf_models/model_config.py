"""Functions related to configurations of Hugging Face models."""

from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils.endpoint_helpers import ModelFilter
from requests.exceptions import RequestException

from ..config import ModelConfig
from ..enums import Framework
from ..exceptions import HuggingFaceHubDown, InvalidBenchmark, NoInternetConnection
from ..languages import get_all_languages
from ..utils import internet_connection_available


def get_hf_model_config(model_id: str, use_auth_token: bool | str) -> ModelConfig:
    """Fetches configuration for a model from the Hugging Face Hub.

    Args:
        model_id (str):
            The full Hugging Face Hub ID of the model.
        use_auth_token (bool or str):
            The Hugging Face Hub authentication token, or `True` to use the stored
            credentials through the `huggingface-cli` package.

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
            use_auth_token=use_auth_token,
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
