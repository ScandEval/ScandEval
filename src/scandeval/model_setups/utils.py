"""Utility functions related to setting up models."""

from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..exceptions import InvalidModel
from ..utils import DUMMY_FILL_VALUE, model_is_generative


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


def setup_model_for_question_answering(model: PreTrainedModel) -> PreTrainedModel:
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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_length: int,
    raise_errors: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Aligns the model and the tokenizer.

    Args:
        model:
            The model to fix.
        tokenizer:
            The tokenizer to fix.
        generation_length:
            The length of the generation, which depends on the benchmark dataset. Only
            relevant if the model is a generative model.
        raise_errors:
            Whether to raise errors instead of trying to fix them silently.

    Returns:
        The fixed model and tokenizer.
    """
    # Get all possible maximal lengths
    all_max_lengths: list[int] = []

    # Add the registered max length of the tokenizer
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < 100_000:
        all_max_lengths.append(tokenizer.model_max_length)

    # Add the max length derived from the position embeddings
    if hasattr(model.config, "max_position_embeddings"):
        all_max_lengths.append(model.config.max_position_embeddings)

    # Add the max length derived from the model's input sizes
    if hasattr(tokenizer, "max_model_input_sizes"):
        all_max_lengths.extend(
            [
                size
                for size in tokenizer.max_model_input_sizes.values()
                if size is not None
            ]
        )

    # If the model is a generative model then we need to subtract the generation length
    # from the maximum length, to allow it to keep generating
    if model_is_generative(model=model):
        all_max_lengths = [
            max_length - generation_length for max_length in all_max_lengths
        ]

    # If any maximal lengths were found then use the shortest one
    if len(list(all_max_lengths)) > 0:
        min_max_length = min(list(all_max_lengths))
        tokenizer.model_max_length = min_max_length

    # Otherwise, use the default maximal length
    else:
        tokenizer.model_max_length = 512

    # Manually check that this model max length is valid for the model, and adjust
    # otherwise
    initial_max_length = tokenizer.model_max_length
    for max_length in range(initial_max_length, 0, -1):
        tokenizer.model_max_length = max_length
        dummy_inputs = torch.full(
            size=(1, max_length),
            fill_value=DUMMY_FILL_VALUE,
            dtype=torch.long,
            device=model.device,
        )
        with torch.inference_mode():
            try:
                model(dummy_inputs)
                break

            # This handles the case where the model is a sequence-to-sequence model, as
            # they require text labels to be passed in
            except ValueError as e:
                if "decoder_input_ids" not in str(e):
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
            model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + 1)

    # For generative models, the `transformers` package requires the pad token to be
    # identical to the eos token, if the latter exists. Otherwise, if both the pad and
    # eos token are not defined, then we attempt to set the padding token to the sep
    # token. If a sep token doesn't exist either, we raise an error.
    if model_is_generative(model=model):
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

    return model, tokenizer
