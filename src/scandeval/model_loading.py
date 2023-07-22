"""Functions related to the loading of models."""

from typing import Type

from transformers import PreTrainedModel

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .model_setups import (
    FreshModelSetup,
    GenerativeModel,
    HFModelSetup,
    LocalModelSetup,
    ModelSetup,
    OpenAIModelSetup,
    Tokenizer,
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
    tokenizer, model = setup.load_model(
        model_config=model_config, dataset_config=dataset_config
    )

    # TODO: XMOD model setup: https://huggingface.co/facebook/xmod-base#input-language

    return tokenizer, model
