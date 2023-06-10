"""Functions related to the existence of Hugging Face models."""

from huggingface_hub.hf_api import HfApi, ModelInfo
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import RequestException

from ..exceptions import HuggingFaceHubDown, NoInternetConnection
from ..utils import internet_connection_available


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
