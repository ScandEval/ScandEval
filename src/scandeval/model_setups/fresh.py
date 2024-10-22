"""Model setup for fresh models."""

import re
from json import JSONDecodeError
from typing import TYPE_CHECKING

from transformers import (
    AutoConfig,
    AutoTokenizer,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
)

from ..config import ModelConfig
from ..enums import Framework, ModelType
from ..exceptions import InvalidBenchmark, InvalidModel
from ..utils import block_terminal_output, create_model_cache_dir, model_is_generative
from .utils import align_model_and_tokenizer, setup_model_for_question_answering

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

    from ..config import BenchmarkConfig, DatasetConfig
    from ..protocols import GenerativeModel, Tokenizer


FRESH_MODELS: list[str] = ["electra-small", "xlm-roberta-base"]


class FreshModelSetup:
    """Model setup for fresh models.

    Attributes:
        benchmark_config:
            The benchmark configuration.
    """

    def __init__(self, benchmark_config: "BenchmarkConfig") -> None:
        """Initialize the FreshModelSetup class.

        Args:
            benchmark_config:
                The benchmark configuration.
        """
        self.benchmark_config = benchmark_config

    @staticmethod
    def _strip_model_id(model_id: str) -> str:
        return re.sub("(@.*$|^fresh-)", "", model_id)

    def model_exists(self, model_id: str) -> bool | dict[str, str]:
        """Check if a model ID denotes a fresh model.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exist, or a dictionary explaining why we cannot check
            whether the model exists.
        """
        return self._strip_model_id(model_id=model_id) in FRESH_MODELS

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Fetches configuration for a fresh model.

        Args:
            model_id:
                The model ID.

        Returns:
            The model configuration.
        """
        return ModelConfig(
            model_id=self._strip_model_id(model_id=model_id),
            framework=Framework.PYTORCH,
            task="fill-mask",
            languages=list(),
            revision="main",
            model_type=ModelType.FRESH,
            model_cache_dir=create_model_cache_dir(
                cache_dir=self.benchmark_config.cache_dir, model_id=model_id
            ),
        )

    def load_model(
        self, model_config: ModelConfig, dataset_config: "DatasetConfig"
    ) -> tuple["PreTrainedModel | GenerativeModel", "Tokenizer"]:
        """Load a fresh model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.

        Returns:
            The tokenizer and model.
        """
        config: "PretrainedConfig"
        block_terminal_output()

        model_id = model_config.model_id
        supertask = dataset_config.task.supertask

        if model_config.model_id == "xlm-roberta-base":
            if supertask == "sequence-classification":
                model_cls = XLMRobertaForSequenceClassification
            elif supertask == "token-classification":
                model_cls = XLMRobertaForTokenClassification
            elif supertask == "question-answering":
                model_cls = XLMRobertaForQuestionAnswering
            else:
                raise InvalidBenchmark(
                    f"Supertask {supertask} is not supported for model {model_id}"
                )

        elif model_id == "electra-small":
            model_id = "google/electra-small-discriminator"
            if supertask == "sequence-classification":
                model_cls = ElectraForSequenceClassification
            elif supertask == "token-classification":
                model_cls = ElectraForTokenClassification
            elif supertask == "question-answering":
                model_cls = ElectraForQuestionAnswering
            else:
                raise InvalidBenchmark(
                    f"Supertask {supertask} is not supported for model {model_id}"
                )

        else:
            raise InvalidModel(f"Model {model_id} is not supported as a fresh class.")

        config = AutoConfig.from_pretrained(
            model_id,
            token=self.benchmark_config.token,
            num_labels=dataset_config.num_labels,
            id2label=dataset_config.id2label,
            label2id=dataset_config.label2id,
            cache_dir=model_config.model_cache_dir,
        )
        model = model_cls(config)

        if supertask == "question-answering":
            model = setup_model_for_question_answering(model=model)

        # Load the tokenizer. If the model is a subclass of a RoBERTa model then we
        # have to add a prefix space to the tokens, by the way the model is constructed
        prefix_models = ["Roberta", "GPT", "Deberta"]
        prefix = any(model_type in type(model).__name__ for model_type in prefix_models)
        try:
            tokenizer: "PreTrainedTokenizerBase" = AutoTokenizer.from_pretrained(
                model_id,
                revision=model_config.revision,
                token=self.benchmark_config.token,
                add_prefix_space=prefix,
                cache_dir=model_config.model_cache_dir,
                use_fast=True,
                verbose=False,
            )
        except (JSONDecodeError, OSError):
            raise InvalidModel(f"Could not load tokenizer for model {model_id!r}.")

        model, tokenizer = align_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            generation_length=dataset_config.max_generated_tokens,
            raise_errors=self.benchmark_config.raise_errors,
            generative_model=model_is_generative(model=model),
        )

        return model, tokenizer
