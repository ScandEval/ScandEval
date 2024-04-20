"""Functions related to the loading of models."""

from typing import TYPE_CHECKING, Type

from .exceptions import InvalidBenchmark
from .model_setups import (
    FreshModelSetup,
    HFModelSetup,
    LocalModelSetup,
    OpenAIModelSetup,
)
from .utils import (
    GENERATIVE_DATASET_SUPERTASKS,
    GENERATIVE_DATASET_TASKS,
    model_is_generative,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .config import BenchmarkConfig, DatasetConfig, ModelConfig
    from .protocols import GenerativeModel, ModelSetup, Tokenizer


def load_model(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> tuple["PreTrainedModel | GenerativeModel", "Tokenizer"]:
    """Load a model.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The tokenizer and model.
    """
    model_type_to_model_setup_mapping: dict[str, Type["ModelSetup"]] = dict(
        fresh=FreshModelSetup,
        hf=HFModelSetup,
        local=LocalModelSetup,
        openai=OpenAIModelSetup,
    )
    setup_class = model_type_to_model_setup_mapping[model_config.model_type]
    setup = setup_class(benchmark_config=benchmark_config)

    model, tokenizer = setup.load_model(
        model_config=model_config, dataset_config=dataset_config
    )

    # Refuse to benchmark non-generative models on generative tasks
    if (
        (
            dataset_config.task.supertask in GENERATIVE_DATASET_SUPERTASKS
            or dataset_config.task.name in GENERATIVE_DATASET_TASKS
        )
        and model is not None
        and not model_is_generative(model=model)
    ):
        raise InvalidBenchmark(
            f"Cannot benchmark non-generative model {model_config.model_id!r} on "
            f"generative task {dataset_config.task.name!r}."
        )

    # TODO: XMOD model setup: https://huggingface.co/facebook/xmod-base#input-language

    return model, tokenizer
