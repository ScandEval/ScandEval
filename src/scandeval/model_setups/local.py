"""Model setup for local Hugging Face Hub models."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from ..config import ModelConfig
from ..enums import Framework, ModelType
from ..exceptions import InvalidModel
from ..utils import create_model_cache_dir
from .hf import HFModelSetup

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..config import BenchmarkConfig, DatasetConfig
    from ..protocols import GenerativeModel, Tokenizer


logger = logging.getLogger(__package__)


class LocalModelSetup:
    """Model setup for local Hugging Face Hub models.

    Attributes:
        benchmark_config:
            The benchmark configuration.
    """

    def __init__(self, benchmark_config: "BenchmarkConfig") -> None:
        """Initialize the LocalModelSetup class.

        Args:
            benchmark_config:
                The benchmark configuration.
        """
        self.benchmark_config = benchmark_config

    def model_exists(self, model_id: str) -> bool | dict[str, str]:
        """Check if a model exists locally.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exist, or a dictionary explaining why we cannot check
            whether the model exists.
        """
        model_dir = Path(model_id)

        if not model_dir.exists():
            return False

        try:
            AutoConfig.from_pretrained(model_id)
        except OSError:
            return False

        return (
            model_dir.glob("*.bin") is not None
            or model_dir.glob("*.pt") is not None
            or model_dir.glob("*.msgpack") is not None
            or model_dir.glob("*.safetensors") is not None
        )

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Fetches configuration for a local Hugging Face model.

        Args:
            model_id:
                The model ID of the model.

        Returns:
            The model configuration.
        """
        framework = self.benchmark_config.framework
        if framework is None:
            try:
                exts = {f.suffix for f in Path(model_id).iterdir()}
                if ".bin" in exts:
                    framework = Framework.PYTORCH
                elif ".msgpack" in exts:
                    framework = Framework.JAX
                elif ".whl" in exts:
                    raise InvalidModel("SpaCy models are not supported.")
                elif ".h5" in exts:
                    raise InvalidModel("TensorFlow/Keras models are not supported.")
            except OSError as e:
                logger.info(f"Cannot list files for local model `{model_id}`!")
                if self.benchmark_config.raise_errors:
                    raise e

        if framework is None:
            msg = f"Assuming 'pytorch' as the framework for local model `{model_id}`. "
            if self.benchmark_config.run_with_cli:
                msg += (
                    "If this is not the case then please use the --framework argument "
                    "to override."
                )
            else:
                msg += (
                    "If this is not the case then please use the `framework` argument "
                    "in the `Benchmarker` class to override."
                )
            logger.info(msg)
            framework = Framework.PYTORCH

        hf_model_config = AutoConfig.from_pretrained(model_id)
        model_type = hf_model_config.model_type.lower()
        if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            task = "text-generation"
        elif model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            task = "text2text-generation"
        elif model_type in MODEL_FOR_MASKED_LM_MAPPING_NAMES:
            task = "fill-mask"
        else:
            task = "unknown"

        model_config = ModelConfig(
            model_id=model_id,
            revision="main",
            framework=framework,
            task=task,
            languages=list(),
            model_type=ModelType.LOCAL,
            model_cache_dir=create_model_cache_dir(
                cache_dir=self.benchmark_config.cache_dir, model_id=model_id
            ),
        )
        return model_config

    def load_model(
        self, model_config: ModelConfig, dataset_config: "DatasetConfig"
    ) -> tuple["PreTrainedModel | GenerativeModel", "Tokenizer"]:
        """Load a local Hugging Face model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.

        Returns:
            The tokenizer and model.
        """
        hf_model_setup = HFModelSetup(benchmark_config=self.benchmark_config)
        return hf_model_setup.load_model(
            model_config=model_config, dataset_config=dataset_config
        )
