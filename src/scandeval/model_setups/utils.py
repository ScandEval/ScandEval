"""Utility functions related to setting up models."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..exceptions import InvalidModel
from ..utils import DUMMY_FILL_VALUE, get_model_max_length
from ..vllm_models import VLLMModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_children_of_module(
    name: str, module: nn.Module
) -> nn.Module | dict[str, Any] | None:
    """Get the children of a module.

    Args:
        name:
            The name of the module.
        module:
            The module to get the children of.

    Returns:
        The children of the module, or None if the module has no children.
    """
    if len(list(module.children())) == 0:
        if name == "token_type_embeddings":
            return module
        else:
            return None
    else:
        submodules = dict()
        for subname, submodule in module.named_children():
            children = get_children_of_module(name=subname, module=submodule)
            if children:
                submodules[subname] = children
        return submodules


def setup_model_for_question_answering(model: "PreTrainedModel") -> "PreTrainedModel":
    """Setup a model for question answering.

    Args:
        model:
            The model to setup.

    Returns:
        The setup model.
    """
    # Get the models' token type embedding children, if they exist
    children = get_children_of_module(name="model", module=model)

    # If the model has token type embeddings then get them
    if children:
        # Get the list of attributes that are token type embeddings
        attribute_list = list()
        done = False
        while not done:
            for key, value in children.items():
                attribute_list.append(key)
                if isinstance(value, dict):
                    children = value
                else:
                    done = True
                break

        # Get the token type embeddings
        token_type_embeddings = model
        for attribute in attribute_list:
            token_type_embeddings = getattr(token_type_embeddings, attribute)

        # If the token type embeddings has shape (1, ...) then set the shape to
        # (2, ...) by randomly initializing the second token type embedding
        if token_type_embeddings.weight.data.shape[0] == 1:
            token_type_embeddings.weight.data = torch.cat(
                (
                    token_type_embeddings.weight.data,
                    torch.rand_like(token_type_embeddings.weight.data),
                ),
                dim=0,
            )
            token_type_embeddings.num_embeddings = 2

        # Set the model config to use the new type vocab size
        model.config.type_vocab_size = 2

    return model


def align_model_and_tokenizer(
    model: "PreTrainedModel | VLLMModel",
    tokenizer: "PreTrainedTokenizerBase",
    generative_model: bool,
    generation_length: int,
    raise_errors: bool = False,
) -> tuple["PreTrainedModel | VLLMModel", "PreTrainedTokenizerBase"]:
    """Aligns the model and the tokenizer.

    Args:
        model:
            The model to fix.
        tokenizer:
            The tokenizer to fix.
        generative_model:
            Whether the model is a generative model.
        generation_length:
            The length of the generation, which depends on the benchmark dataset. Only
            relevant if the model is a generative model.
        raise_errors:
            Whether to raise errors instead of trying to fix them silently.

    Returns:
        The fixed model and tokenizer.
    """
    model_max_length = get_model_max_length(model=model, tokenizer=tokenizer)

    # If the model is a generative model then we need to subtract the generation length
    # from the maximum length, to allow it to keep generating
    if generative_model:
        model_max_length -= generation_length

    # Ensure that the model max length is at least 5,000, to avoid OOM errors
    model_max_length = min(model_max_length, 5_000)

    if model_max_length > 0:
        tokenizer.model_max_length = model_max_length
    elif generative_model:
        tokenizer.model_max_length = 5_000
    else:
        tokenizer.model_max_length = 512

    # If we're not dealing with a generative model then we move it to CPU to avoid OOM
    # errors
    device = model.device if generative_model else torch.device("cpu")
    model_device = model.device
    model.to(device)

    # Manually check that this model max length is valid for the model, and adjust
    # otherwise
    initial_max_length = tokenizer.model_max_length
    for max_length in range(initial_max_length, 0, -1):
        tokenizer.model_max_length = max_length
        dummy_inputs = torch.full(
            size=(1, max_length),
            fill_value=DUMMY_FILL_VALUE,
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            try:
                model(dummy_inputs)
                break

            # This handles the case where the model is a sequence-to-sequence model, as
            # they require text labels to be passed in
            except ValueError as e:
                if "decoder_input_ids" not in str(e) or isinstance(model, VLLMModel):
                    raise e
                model(input_ids=dummy_inputs, labels=torch.zeros(1, 1).long())
                break

            # This happens if `max_length` is too large
            except IndexError:
                continue

    # If there is a mismatch between the vocab size according to the tokenizer and
    # the vocab size according to the model, we raise an error
    if hasattr(model.config, "vocab_size") and hasattr(tokenizer, "vocab_size"):
        if model.config.vocab_size < tokenizer.vocab_size:
            if raise_errors:
                raise InvalidModel(
                    "The vocab size of the tokenizer is larger than the vocab size of "
                    "the model. As the --raise-errors option was specified, the "
                    "embeddings of the model will not be automatically adjusted."
                )
            if isinstance(model, VLLMModel):
                raise InvalidModel(
                    "The vocab size of the tokenizer is larger than the vocab size of "
                    "the model."
                )
            model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + 1)

    # For generative models, the `transformers` package requires the pad token to be
    # identical to the eos token, if the latter exists. Otherwise, if both the pad and
    # eos token are not defined, then we attempt to set the padding token to the sep
    # token. If a sep token doesn't exist either, we raise an error.
    if generative_model:
        tokenizer.padding_side = "left"
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.pad_token is None:
            if tokenizer.sep_token is not None:
                tokenizer.pad_token = tokenizer.sep_token
            else:
                raise InvalidModel(
                    "The tokenizer does not have a padding token and does not have a "
                    "SEP token or EOS token to use as a padding token."
                )
        model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.bos_token is None and tokenizer.eos_token is not None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    model.to(model_device)

    return model, tokenizer
