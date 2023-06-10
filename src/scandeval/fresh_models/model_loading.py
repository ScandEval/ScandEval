"""Functions related to the loading of fresh Hugging Face models."""

import warnings
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

from ..exceptions import InvalidBenchmark
from ..hf_models.model_loading import (
    align_model_and_tokenizer,
    setup_model_for_question_answering,
)
from ..utils import block_terminal_output


def load_fresh_model(
    model_id: str,
    revision: str,
    supertask: str,
    num_labels: int,
    id2label: list[str],
    label2id: dict[str, int],
    use_auth_token: bool | str,
    cache_dir: str,
    raise_errors: bool = False,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a fresh model.

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
        use_auth_token (bool or str):
            Whether to use an authentication token to access the model. If a boolean
            value is specified then it is assumed that the user is logged in to the
            Hugging Face CLI, and if a string is specified then it is used as the
            token.
        cache_dir (str):
            The directory to cache the model in.
        raise_errors (bool, optional):
            Whether to raise errors instead of trying to fix them silently.

    Returns:
        pair of (tokenizer, model):
            The tokenizer and model.

    Raises:
        RuntimeError:
            If the framework is not recognized.
    """
    config: PretrainedConfig
    block_terminal_output()

    if model_id == "xlm-roberta-base":
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
        raise InvalidBenchmark(f"Model {model_id} is not supported as a fresh class.")

    config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=use_auth_token,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )
    model = model_cls(config)

    # Set up the model for question answering
    if supertask == "question-answering":
        model = setup_model_for_question_answering(model=model)

    # Load the tokenizer. If the model is a subclass of a RoBERTa model then we have to
    # add a prefix space to the tokens, by the way the model is constructed.
    prefix_models = ["Roberta", "GPT", "Deberta"]
    prefix = any(model_type in type(model).__name__ for model_type in prefix_models)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                use_auth_token=use_auth_token,
                add_prefix_space=prefix,
                cache_dir=cache_dir,
                use_fast=True,
                verbose=False,
            )
        except (JSONDecodeError, OSError):
            raise InvalidBenchmark(f"Could not load tokenizer for model {model_id!r}.")

    # Align the model and the tokenizer
    model, tokenizer = align_model_and_tokenizer(
        model=model, tokenizer=tokenizer, raise_errors=raise_errors
    )

    return tokenizer, model
