"""Functions related to getting the model configuration."""

import importlib.util
from typing import TYPE_CHECKING

from .enums import Framework
from .exceptions import InvalidModel, NeedsExtraInstalled
from .model_setups import MODEL_SETUP_CLASSES

if TYPE_CHECKING:
    from .config import BenchmarkConfig, ModelConfig


def get_model_config(
    model_id: str, benchmark_config: "BenchmarkConfig"
) -> "ModelConfig":
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
    needs_env_vars: list[str] = list()
    for setup_class in MODEL_SETUP_CLASSES:
        setup = setup_class(benchmark_config=benchmark_config)

        exists_or_dict = setup.model_exists(model_id=model_id)
        if isinstance(exists_or_dict, dict):
            if "missing_extra" in exists_or_dict:
                needs_extras.append(exists_or_dict["missing_extra"])
            elif "missing_env_var" in exists_or_dict:
                needs_env_vars.append(exists_or_dict["missing_env_var"])
            else:
                raise ValueError(
                    "The dictionary returned by `model_exists` must contain either "
                    "the key `missing_extra` or `missing_env_var`."
                )
        elif exists_or_dict:
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
        elif needs_env_vars:
            msg += (
                " However, it is possible that the model exists, but an environment "
                "variable needs to be set to check if it exists. Please set the "
                f"environment variables {','.join(needs_env_vars)} and try again."
            )
        raise InvalidModel(msg)
