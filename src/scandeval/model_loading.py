"""Functions related to the loading of models."""

from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .model_setups import MODEL_SETUP_CLASSES


def load_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a model.

    Args:
        model_config (ModelConfig):
            The model configuration.
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.

    Returns:
        pair of (tokenizer, model):
            The tokenizer and model.

    Raises:
        RuntimeError:
            If the model doesn't exist.
    """
    model_id = model_config.model_id
    for setup_class in MODEL_SETUP_CLASSES:
        setup = setup_class(benchmark_config=benchmark_config)
        if setup.model_exists(model_id=model_id):
            return setup.load_model(
                model_config=model_config,
                dataset_config=dataset_config,
            )
    else:
        raise RuntimeError(f"Model {model_id} not found.")
