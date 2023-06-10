"""Functions related to the loading of models."""

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .enums import Framework
from .fresh_models import load_fresh_model, model_exists_fresh
from .hf_models import load_hf_model, model_exists_on_hf_hub
from .local_models import load_local_model, model_exists_locally
from .openai_models import load_openai_model, model_exists_on_openai


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
            If the framework is not recognized.
    """
    model_id = model_config.model_id
    if model_exists_fresh(model_id=model_id):
        return load_fresh_model(
            model_id=model_id,
            revision=model_config.revision,
            supertask=dataset_config.task.supertask,
            num_labels=dataset_config.num_labels,
            id2label=dataset_config.id2label,
            label2id=dataset_config.label2id,
            use_auth_token=benchmark_config.use_auth_token,
            cache_dir=benchmark_config.cache_dir,
            raise_errors=benchmark_config.raise_errors,
        )
    elif model_exists_locally(model_id=model_id):
        return load_local_model(
            model_id=model_id,
            revision=model_config.revision,
            supertask=dataset_config.task.supertask,
            language=dataset_config.languages[0].code,
            num_labels=dataset_config.num_labels,
            id2label=dataset_config.id2label,
            label2id=dataset_config.label2id,
            from_flax=model_config.framework == Framework.JAX,
            use_auth_token=benchmark_config.use_auth_token,
            cache_dir=benchmark_config.cache_dir,
            raise_errors=benchmark_config.raise_errors,
        )
    elif model_exists_on_hf_hub(
        model_id=model_id, use_auth_token=benchmark_config.use_auth_token
    ):
        return load_hf_model(
            model_id=model_id,
            revision=model_config.revision,
            supertask=dataset_config.task.supertask,
            language=dataset_config.languages[0].code,
            num_labels=dataset_config.num_labels,
            id2label=dataset_config.id2label,
            label2id=dataset_config.label2id,
            from_flax=model_config.framework == Framework.JAX,
            use_auth_token=benchmark_config.use_auth_token,
            cache_dir=benchmark_config.cache_dir,
            raise_errors=benchmark_config.raise_errors,
        )
    elif model_exists_on_openai(
        model_id=model_id, openai_api_key=benchmark_config.openai_api_key
    ):
        return load_openai_model(
            model_id=model_id,
            openai_api_key=benchmark_config.openai_api_key,
            cache_dir=benchmark_config.cache_dir,
            raise_errors=benchmark_config.raise_errors,
        )
    else:
        raise RuntimeError(
            f"Model {model_id} not found in any of the supported locations."
        )
