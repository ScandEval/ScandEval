"""Functions related to the loading of models."""

import typing as t

from .benchmark_modules import (
    FreshEncoderModel,
    HuggingFaceEncoderModel,
    LiteLLMModel,
    VLLMModel,
)
from .constants import GENERATIVE_DATASET_SUPERTASKS, GENERATIVE_DATASET_TASKS
from .enums import ModelType
from .exceptions import InvalidBenchmark

if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


def load_model(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> "BenchmarkModule":
    """Load a model.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The model.
    """
    # The order matters; the first model type that matches will be used. For this
    # reason, they have been ordered in terms of the most common model types.
    model_type_to_module_mapping: dict[ModelType, t.Type[BenchmarkModule]] = {
        ModelType.HF_HUB_GENERATIVE: VLLMModel,
        ModelType.HF_HUB_ENCODER: HuggingFaceEncoderModel,
        ModelType.API: LiteLLMModel,
        ModelType.FRESH: FreshEncoderModel,
    }
    model_class = model_type_to_module_mapping[model_config.model_type]

    model = model_class(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )

    # Refuse to benchmark non-generative models on generative tasks
    if (
        (
            dataset_config.task.supertask in GENERATIVE_DATASET_SUPERTASKS
            or dataset_config.task.name in GENERATIVE_DATASET_TASKS
        )
        and model is not None
        and not model.is_generative
    ):
        raise InvalidBenchmark(
            f"Cannot benchmark non-generative model {model_config.model_id!r} on "
            f"generative task {dataset_config.task.name!r}."
        )

    return model
