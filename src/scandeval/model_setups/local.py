"""Model setup for local Hugging Face Hub models."""

import logging
from pathlib import Path

from transformers import AutoConfig, PreTrainedModel

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig
from ..enums import Framework, ModelType
from ..exceptions import InvalidModel
from ..protocols import GenerativeModel, Tokenizer
from ..utils import create_model_cache_dir
from .hf import HFModelSetup

logger = logging.getLogger(__package__)


class LocalModelSetup:
    """Model setup for local Hugging Face Hub models.

    Attributes:
        benchmark_config:
            The benchmark configuration.
    """

    def __init__(self, benchmark_config: BenchmarkConfig) -> None:
        """Initialize the LocalModelSetup class.

        Args:
            benchmark_config:
                The benchmark configuration.
        """
        self.benchmark_config = benchmark_config

    def model_exists(self, model_id: str) -> bool | str:
        """Check if a model exists locally.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exists locally, or the name of an extra that needs to be
            installed to check if the model exists.
        """
        # Ensure that `model_id` is a Path object
        model_dir = Path(model_id)

        # Return False if the model folder does not exist
        if not model_dir.exists():
            return False

        # Try to load the model config. If this fails, False is returned
        try:
            AutoConfig.from_pretrained(model_id)
        except OSError:
            return False

        return (
            model_dir.glob("*.bin") is not None
            or model_dir.glob("*.pt") is not None
            or model_dir.glob("*.msgpack") is not None
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
            logger.info(
                f"Assuming 'pytorch' as the framework for local model `{model_id}`! "
                "If this is in error, please use the --framework option to override."
            )
            framework = Framework.PYTORCH

        hf_model_config = AutoConfig.from_pretrained(model_id)
        model_type = hf_model_config.model_type.lower()
        if "gpt" in model_type:
            task = "text-generation"
        elif "t5" in model_type or "bart" in model_type:
            task = "text2text-generation"
        elif "bert" in model_type:
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
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> tuple[Tokenizer, PreTrainedModel | GenerativeModel]:
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
