"""Functions related to the loading of OpenAI models."""


def load_openai_model(
    model_id: str,
    openai_api_key: str | None,
    cache_dir: str,
    raise_errors: bool = False,
):
    """Load an OpenAI model.

    Args:
        model_id (str):
            The Hugging Face ID of the model.
        openai_api_key (str or None):
            The OpenAI API key. If None, the environment variable `OPENAI_API_KEY` is
            used.
        cache_dir (str):
            The directory to cache the model in.
        raise_errors (bool, optional):
            Whether to raise errors instead of trying to fix them silently.

    Returns:
        TODO
    """
    raise NotImplementedError
