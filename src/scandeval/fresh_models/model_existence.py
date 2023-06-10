"""Functions related to the existence of fresh Hugging Face models."""

import re

FRESH_MODELS: list[str] = [
    "electra-small",
    "xlm-roberta-base",
]


def model_exists_fresh(model_id: str) -> bool:
    """Check if a model ID denotes a fresh model.

    Args:
        model_id (str):
            The model ID.

    Returns:
        bool:
            Whether the model exists locally.
    """
    # Remove model type and revision from model ID
    model_id = re.sub("(^.*::|@.*$|^fresh-)", "", model_id)

    return model_id in FRESH_MODELS
