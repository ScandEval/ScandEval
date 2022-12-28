"""Functions related to the loading of models."""

import warnings
from typing import Dict, List, Tuple, Type, Union

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.electra.modeling_electra import (
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from .exceptions import InvalidBenchmark
from .utils import block_terminal_output, get_class_by_name


def load_model(
    model_id: str,
    revision: str,
    supertask: str,
    num_labels: int,
    id2label: List[str],
    label2id: Dict[str, int],
    from_flax: bool,
    use_auth_token: Union[bool, str],
    cache_dir: str,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a model.

    Args:
        model_id (str):
            The Hugging Face ID of the model.
        revision (str):
            The specific version of the model. Can be a branch name, a tag name, or a
            commit hash.
        supertask (str):
            The supertask of the task to benchmark the model on.
        num_labels (int):
            The number of labels in the dataset.
        id2label (list of str):
            The mapping from ID to label.
        label2id (dict of str to int):
            The mapping from label to ID.
        from_flax (bool):
            Whether the model is a Flax model.
        use_auth_token (bool or str):
            Whether to use an authentication token to access the model. If a boolean
            value is specified then it is assumed that the user is logged in to the
            Hugging Face CLI, and if a string is specified then it is used as the
            token.
        cache_dir (str):
            The directory to cache the model in.

    Returns:
        pair of (tokenizer, model):
            The tokenizer and model.

    Raises:
        RuntimeError:
            If the framework is not recognized.
    """
    config: PretrainedConfig
    block_terminal_output()

    try:
        # If the model ID specifies a fresh model, then load that.
        if model_id.startswith("fresh"):
            model_cls, model_id = load_fresh_model_class(
                model_id=model_id, supertask=supertask
            )
            config = AutoConfig.from_pretrained(
                model_id,
                use_auth_token=use_auth_token,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                cache_dir=cache_dir,
            )
            model = model_cls(config)

        # Otherwise load the pretrained model
        else:
            config = AutoConfig.from_pretrained(
                model_id,
                revision=revision,
                use_auth_token=use_auth_token,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                cache_dir=cache_dir,
            )

            # Get the model class associated with the supertask
            model_cls_or_none: Union[None, Type[PreTrainedModel]] = get_class_by_name(
                class_name=f"auto-model-for-{supertask}",
                module_name="transformers",
            )

            # If the model class could not be found then raise an error
            if not model_cls_or_none:
                raise InvalidBenchmark(
                    f"The supertask '{supertask}' does not correspond to a Hugging Face"
                    " AutoModel type (such as `AutoModelForSequenceClassification`)."
                )

            # If the model is a DeBERTaV2 model then we ensure that
            # `pooler_hidden_size` is the same size as `hidden_size`
            if config.model_type == "deberta-v2":
                config.pooler_hidden_size = config.hidden_size

            # If we are benchmarking a question answering model then we need to set
            # the type_vocab_size to 2
            if supertask == "question-answering":
                config.type_vocab_size = 2

            # Load the model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model_or_tuple = model_cls_or_none.from_pretrained(
                    model_id,
                    revision=revision,
                    use_auth_token=use_auth_token,
                    config=config,
                    cache_dir=cache_dir,
                    from_flax=from_flax,
                    ignore_mismatched_sizes=True,
                )
            if isinstance(model_or_tuple, tuple):
                model = model_or_tuple[0]
            else:
                model = model_or_tuple

    except (OSError, ValueError) as e:

        # Deal with the case where the checkpoint is incorrect
        if "checkpoint seems to be incorrect" in str(e):
            raise InvalidBenchmark(
                f"The model {model_id!r} has an incorrect checkpoint."
            )

        # Otherwise raise a more generic error
        raise InvalidBenchmark(
            f"The model {model_id} either does not exist on the Hugging Face Hub, or "
            "it has no frameworks registered, or it is a private model. If it *does* "
            "exist on the Hub and is a public model then please ensure that it has a "
            "framework registered. If it is a private model then enable the "
            "`--use-auth-token` flag and make sure that you are logged in to the Hub "
            "via the `huggingface-cli login` command."
        )

    # If the model is a subclass of a RoBERTa model then we have to add a prefix
    # space to the tokens, by the way the model is constructed.
    prefix_models = ["Roberta", "GPT", "Deberta"]
    prefix = any(model_type in type(model).__name__ for model_type in prefix_models)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_auth_token=use_auth_token,
            add_prefix_space=prefix,
            cache_dir=cache_dir,
            use_fast=True,
            verbose=False,
        )

    # Fix the model and the tokenizer
    model, tokenizer = fix_model_and_tokenizer(model, tokenizer)

    return tokenizer, model


def load_fresh_model_class(
    model_id: str, supertask: str
) -> Tuple[Type[PreTrainedModel], str]:
    """Load a fresh model class.

    Args:
        model_id (str):
            The Hugging Face ID of the model.
        supertask (str):
            The supertask of the task to benchmark the model on.

    Returns:
        pair of Type[PreTrainedModel] and str:
            The model class and the pretrained model ID.
    """
    if model_id == "fresh-xlmr-base":
        model_id = "xlm-roberta-base"
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

    elif model_id == "fresh-electra-small":
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
        raise InvalidBenchmark(
            f"A fresh model was chosen, `{model_id}`, but it was not recognized."
        )

    return model_cls, model_id


def fix_model_and_tokenizer(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Fixes the model and tokenizer to be compatible with the benchmarking framework.

    Args:
        model (PreTrainedModel):
            The model to fix.
        tokenizer (PreTrainedTokenizer):
            The tokenizer to fix.

    Returns:
        pair of (model, tokenizer):
            The fixed model and tokenizer.
    """
    # Set the maximal length of the tokenizer to the model's maximal length. This is
    # required for proper truncation
    if (
        not hasattr(tokenizer, "model_max_length")
        or tokenizer.model_max_length > 100_000
    ):

        if hasattr(tokenizer, "max_model_input_sizes"):
            all_max_lengths = tokenizer.max_model_input_sizes.values()
            if len(list(all_max_lengths)) > 0:
                min_max_length = min(list(all_max_lengths))
                tokenizer.model_max_length = min_max_length
            else:
                tokenizer.model_max_length = 512
        else:
            tokenizer.model_max_length = 512

    # If there is a mismatch between the vocab size according to the tokenizer and
    # the vocab size according to the model, we raise an error
    if hasattr(model.config, "vocab_size") and hasattr(tokenizer, "vocab_size"):
        if model.config.vocab_size < tokenizer.vocab_size:
            raise InvalidBenchmark(
                "The vocab size of the tokenizer is larger than the vocab size of the "
                "model. This is not supported."
            )

    # If the tokenizer does not have a padding token (e.g. GPT-2), we use find a
    # suitable padding token and set it
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.sep_token is not None:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.sep_token
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise InvalidBenchmark(
                "The tokenizer does not have a padding token and does not have a "
                "SEP token or EOS token to use as a padding token."
            )

    return model, tokenizer
