"""Functions related to the loading of models."""

from typing import Type

from transformers import PreTrainedModel

from .exceptions import InvalidBenchmark
from .utils import GENERATIVE_DATASET_SUPERTASKS
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .protocols import GenerativeModel, Tokenizer, ModelSetup
from .model_setups import (
    FreshModelSetup,
    HFModelSetup,
    LocalModelSetup,
    OpenAIModelSetup,
)


def load_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> tuple[Tokenizer, PreTrainedModel | GenerativeModel]:
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
    model_type_to_model_setup_mapping: dict[str, Type[ModelSetup]] = dict(
        fresh=FreshModelSetup,
        hf=HFModelSetup,
        local=LocalModelSetup,
        openai=OpenAIModelSetup,
    )
    setup_class = model_type_to_model_setup_mapping[model_config.model_type]
    setup = setup_class(benchmark_config=benchmark_config)

    error_message = (
        f"Cannot benchmark non-generative model {model_config.model_id!r} on "
        f"generative task {dataset_config.task.name!r}."
    )

    error_to_raise = None
    model = None
    try:
        tokenizer, model = setup.load_model(
            model_config=model_config, dataset_config=dataset_config
        )
    except InvalidBenchmark as e:
        error_to_raise = e

    # Refuse to benchmark non-generative models on generative tasks
    if (
        dataset_config.task.supertask in GENERATIVE_DATASET_SUPERTASKS
        and not isinstance(model, GenerativeModel)
    ):
        raise InvalidBenchmark(error_message)
    elif error_to_raise is not None:
        raise error_to_raise

    # TODO: XMOD model setup: https://huggingface.co/facebook/xmod-base#input-language

    return tokenizer, model
