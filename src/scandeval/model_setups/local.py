"""Model setup for local Hugging Face Hub models."""

import logging
from pathlib import Path

from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizer

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig
from ..enums import Framework, ModelType
from ..exceptions import InvalidBenchmark
from .hf import HFModelSetup

logger = logging.getLogger(__name__)


class LocalModelSetup:
    """Model setup for local Hugging Face Hub models.

    Args:
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.

    Attributes:
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
    """

    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        self.benchmark_config = benchmark_config

    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists locally.

        Args:
            model_id (str):
                The model ID.

        Returns:
            bool:
                Whether the model exists locally.
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
            model_id (str):
                The model ID of the model.

        Returns:
            ModelConfig:
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
                    raise InvalidBenchmark("SpaCy models are not supported.")
                elif ".h5" in exts:
                    raise InvalidBenchmark("TensorFlow/Keras models are not supported.")
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

        model_config = ModelConfig(
            model_id=model_id,
            revision="main",
            framework=framework,
            task="fill-mask",
            languages=list(),
            model_type=ModelType.LOCAL,
        )
        return model_config

    def load_model(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Load a local Hugging Face model.

        Args:
            model_config (ModelConfig):
                The model configuration.
            dataset_config (DatasetConfig):
                The dataset configuration.

        Returns:
            pair of (tokenizer, model):
                The tokenizer and model.
        """
        hf_model_setup = HFModelSetup(benchmark_config=self.benchmark_config)
        return hf_model_setup.load_model(
            model_config=model_config,
            dataset_config=dataset_config,
        )
