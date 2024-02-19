"""Functions related to getting the model configuration."""

import importlib.util

from .config import BenchmarkConfig, ModelConfig
from .enums import Framework
from .exceptions import InvalidModel, NeedsExtraInstalled
from .model_setups import MODEL_SETUP_CLASSES


def get_model_config(model_id: str, benchmark_config: BenchmarkConfig) -> ModelConfig:
    """Fetches configuration for a model.

    Args:
        model_id:
            The model ID.
        benchmark_config:
            The configuration of the benchmark.

    Returns:
        The model configuration.

    Raises:
        InvalidModel:
            If all model setups can handle the model, but the model does not exist.
    """
    needs_extras: list[str] = list()
    for setup_class in MODEL_SETUP_CLASSES:
        setup = setup_class(benchmark_config=benchmark_config)

        exists_or_missing_extra = setup.model_exists(model_id=model_id)
        if isinstance(exists_or_missing_extra, str):
            needs_extras.append(exists_or_missing_extra)
        elif exists_or_missing_extra:
            model_config = setup.get_model_config(model_id=model_id)
            if (
                model_config.framework == Framework.JAX
                and importlib.util.find_spec("jax") is None
            ):
                raise NeedsExtraInstalled(extra="jax")
            return model_config
    else:
        msg = f"Model {model_id} not found."
        if needs_extras:
            msg += (
                " However, it is possible that the model exists, but a package "
                "needs to be installed to check if it exists. Please try running "
                f"`pip install scandeval[{','.join(needs_extras)}]` or `pip install "
                "scandeval[all]`, and try again."
            )
        raise InvalidModel(msg)
