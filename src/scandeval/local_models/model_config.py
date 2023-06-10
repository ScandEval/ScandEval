"""Functions related to configurations of local Hugging Face models."""

import logging
from pathlib import Path

from ..config import ModelConfig
from ..enums import Framework
from ..exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


def get_local_model_config(
    model_id: str, framework: Framework | str | None, raise_errors: bool
) -> ModelConfig:
    """Builds the configuration for a local model.

    Args:
        model_id (str):
            The path of the local model.
        framework (Framework or str or None):
            The framework of the model, or None to try to infer it automatically.
        raise_errors (bool):
            Whether to raise errors or not.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        OSError:
            If `raise_errors` has been set and the framework cannot be recognized
            automatically and it has not explicitly been provided using `framework`.
    """
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
            if raise_errors:
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
