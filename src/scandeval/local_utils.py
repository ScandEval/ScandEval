"""Utility functions related to local models."""

import logging
from pathlib import Path

from transformers import AutoConfig

from .config import BenchmarkConfig, ModelConfig
from .enums import Framework
from .exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


def get_local_model_config(
    model_id: str, benchmark_config: BenchmarkConfig
) -> ModelConfig:
    """Builds the configuration for a local model.

    Args:
        model_id (str):
            The path of the local model.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        OSError:
            If the --raise-errors option has been set to True and the framework cannot
            be recognized automatically and it has not explicitly been provided using
            --framework.
    """
    framework: Framework | str | None = benchmark_config.framework
    if framework is None:
        try:
            exts = {f.suffix for f in Path(model_id).iterdir()}
            if ".bin" in exts:
                framework = Framework.PYTORCH
            elif ".msgpack" in exts:
                framework = Framework.JAX
            elif ".whl" in exts:
                raise InvalidBenchmark("SpaCy models are not supported.")
            elif ".h5" in exts:
                raise InvalidBenchmark("TensorFlow/Keras models are not supported.")
        except OSError as e:
            logger.info(f"Cannot list files for local model `{model_id}`!")
            if benchmark_config.raise_errors:
                raise e

    if framework is None:
        logger.info(
            f"Assuming 'pytorch' as the framework for local model `{model_id}`! "
            "If this is in error, please use the --framework option to override."
        )
        framework = Framework.PYTORCH

    model_config = ModelConfig(
        model_id=model_id,
        revision="main",
        framework=framework,
        task="fill-mask",
        languages=list(),
    )
    return model_config


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
