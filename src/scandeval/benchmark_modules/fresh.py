"""Freshly initialised encoder models."""

from functools import cached_property
from json import JSONDecodeError

from transformers import (
    AutoConfig,
    AutoTokenizer,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
)

from ..data_models import BenchmarkConfig, ModelConfig
from ..enums import Framework, ModelType
from ..exceptions import (
    InvalidBenchmark,
    InvalidModel,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..utils import block_terminal_output, create_model_cache_dir
from .hf import (
    HuggingFaceEncoderModel,
    align_model_and_tokenizer,
    setup_model_for_question_answering,
)


class FreshEncoderModel(HuggingFaceEncoderModel):
    """A freshly initialised encoder model."""

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        match self.model_config.model_id:
            case "fresh-xlm-roberta-base":
                return 278_885_778
            case "fresh-electra-small":
                return 13_738_755
            case _:
                raise NotImplementedError(
                    f"Number of parameters for model {self.model_config.model_id} is "
                    "not implemented."
                )

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        match self.model_config.model_id:
            case "fresh-xlm-roberta-base":
                return 250_002
            case "fresh-electra-small":
                return 32_000
            case _:
                raise NotImplementedError(
                    f"Vocabulary size for model {self.model_config.model_id} is not "
                    "implemented."
                )

    @cached_property
    def model_max_length(self) -> int:
        """The maximum context length of the model.

        Returns:
            The maximum context length of the model.
        """
        match self.model_config.model_id:
            case "fresh-xlm-roberta-base":
                return 512
            case "fresh-electra-small":
                return 128
            case _:
                raise NotImplementedError(
                    f"Maximum context length for model {self.model_config.model_id} is "
                    "not implemented."
                )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        valid_models = ["fresh-electra-small", "fresh-xlm-roberta-base"]
        return model_id in valid_models

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        return ModelConfig(
            model_id=model_id,
            framework=Framework.PYTORCH,
            task="fill-mask",
            languages=list(),
            revision="main",
            model_type=ModelType.FRESH,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=None,
        )

    def _load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer.

        Returns:
            The loaded model and tokenizer.
        """
        config: "PretrainedConfig"
        block_terminal_output()

        # Get the fresh model ID and the corresponding real model ID
        model_id = self.model_config.model_id.replace("-", "_")
        fresh_to_real_model_id_mapping = dict(
            fresh_xlm_roberta_base="FacebookAI/xlm-roberta-base",
            fresh_electra_small="google/electra-small-discriminator",
        )
        real_model_id = fresh_to_real_model_id_mapping[model_id]

        match self.dataset_config.task.supertask:
            case "sequence-classification":
                model_cls_mapping = dict(
                    fresh_xlm_roberta_base=XLMRobertaForSequenceClassification,
                    fresh_electra_small=ElectraForSequenceClassification,
                )
            case "token-classification":
                model_cls_mapping = dict(
                    fresh_xlm_roberta_base=XLMRobertaForTokenClassification,
                    fresh_electra_small=ElectraForTokenClassification,
                )
            case "question-answering":
                model_cls_mapping = dict(
                    fresh_xlm_roberta_base=XLMRobertaForQuestionAnswering,
                    fresh_electra_small=ElectraForQuestionAnswering,
                )
            case _:
                raise InvalidBenchmark(
                    f"Supertask {self.dataset_config.task.supertask} is not supported "
                    f"for model {self.model_config.model_id}"
                )
        model_cls = model_cls_mapping[model_id]

        config = AutoConfig.from_pretrained(
            real_model_id,
            token=self.benchmark_config.api_key or True,
            num_labels=self.dataset_config.num_labels,
            id2label=self.dataset_config.id2label,
            label2id=self.dataset_config.label2id,
            cache_dir=self.model_config.model_cache_dir,
        )
        model = model_cls(config)

        if self.dataset_config.task.supertask == "question-answering":
            model = setup_model_for_question_answering(model=model)

        # Load the tokenizer. If the model is a subclass of a RoBERTa model then we
        # have to add a prefix space to the tokens, by the way the model is constructed
        prefix_models = ["Roberta", "GPT", "Deberta"]
        prefix = any(model_type in type(model).__name__ for model_type in prefix_models)
        try:
            tokenizer: "PreTrainedTokenizer" = AutoTokenizer.from_pretrained(
                real_model_id,
                revision=self.model_config.revision,
                token=self.benchmark_config.api_key or True,
                add_prefix_space=prefix,
                cache_dir=self.model_config.model_cache_dir,
                use_fast=True,
                verbose=False,
            )
        except (JSONDecodeError, OSError):
            raise InvalidModel(f"Could not load tokenizer for model {real_model_id!r}.")

        model, tokenizer = align_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            raise_errors=self.benchmark_config.raise_errors,
        )

        return model, tokenizer
