"""Functions related to the existence of local Hugging Face models."""

import logging
from pathlib import Path

from transformers import AutoConfig

logger = logging.getLogger(__name__)


def model_exists_locally(model_id: str | Path) -> bool:
    """Check if a Hugging Face model exists locally.

    Args:
        model_id (str or Path):
            Path to the model folder.

    Returns:
        bool:
            Whether the model exists locally.
    """
    # Ensure that `model_id` is a Path object
    model_id = Path(model_id)

    # Return False if the model folder does not exist
    if not model_id.exists():
        return False

    # Try to load the model config. If this fails, False is returned
    try:
        AutoConfig.from_pretrained(str(model_id))
    except OSError:
        return False

    # Check that a compatible model file exists
    pytorch_model_exists = model_id.glob("*.bin") or model_id.glob("*.pt")
    jax_model_exists = model_id.glob("*.msgpack")

    # If no model file exists, return False
    if not pytorch_model_exists and not jax_model_exists:
        return False

    # Otherwise, if all these checks succeeded, return True
    return True
