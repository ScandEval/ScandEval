"""Utility functions related to structured generation."""

import importlib.util
import math
from typing import TYPE_CHECKING, Any, Callable

import torch
from pydantic import BaseModel, conlist, create_model

if importlib.util.find_spec("outlines") is not None:
    from outlines.integrations.transformers import JSONPrefixAllowedTokens
    from outlines.integrations.vllm import JSONLogitsProcessor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM


def get_ner_schema(ner_tag_names: list[str]) -> type[BaseModel]:
    """Get the schema for the NER answer format, used for structured generation.

    Args:
        ner_tag_names:
            The NER tag names.

    Returns:
        The schema for the NER answer format.
    """
    ner_tag_names = sorted(set(ner_tag_names))
    keys_and_their_types: dict[str, Any] = {
        tag_name: (conlist(str, max_length=5), ...) for tag_name in ner_tag_names
    }
    schema = create_model("AnswerFormat", **keys_and_their_types)
    return schema


def get_ner_prefix_allowed_tokens_fn(
    ner_tag_names: list[str], tokenizer: "PreTrainedTokenizerBase"
) -> Callable[[int, torch.Tensor], list[int]]:
    """Get the prefix allowed tokens function for the NER task, used in `transformers`.

    Args:
        ner_tag_names:
            The NER tag names.
        tokenizer:
            The tokenizer to use for tokenizing the JSON Schema.

    Returns:
        The prefix allowed tokens function for the NER task.
    """
    schema = get_ner_schema(ner_tag_names=ner_tag_names)
    json_generation_prefix_allowed_tokens_fn = JSONPrefixAllowedTokens(
        schema=schema, tokenizer_or_pipe=tokenizer, whitespace_pattern=r" ?"
    )

    forbidden_token_ids = list()
    forbidden_tokens = ["\n", "\n\n", "\n\n\n", "\t", "\t\t", "\t\t\t"]
    for forbidden_token in forbidden_tokens:
        forbidden_token_ids.extend(
            list(tokenizer(forbidden_token, add_special_tokens=False).input_ids)
        )
    forbidden_token_ids = list(set(forbidden_token_ids))

    def prefix_allowed_tokens_fn(batch_id: int, sent: torch.Tensor) -> list[int]:
        """Functions used to bias the generation of a `transformers` model.

        Args:
            batch_id:
                The batch ID.
            sent:
                The input tensor.

        Returns:
            The allowed tokens.
        """
        allowed_tokens = json_generation_prefix_allowed_tokens_fn(
            batch_id=batch_id, sent=sent
        )
        return [token for token in allowed_tokens if token not in forbidden_token_ids]

    return prefix_allowed_tokens_fn


def get_ner_logits_processors(
    ner_tag_names: list[str], llm: "LLM"
) -> list[Callable[[list[int], torch.Tensor], torch.Tensor]]:
    """Get the logits processors for the NER task, used in vLLM.

    Args:
        ner_tag_names:
            The NER tag names.
        llm:
            The vLLM model.

    Returns:
        The logit processors for the NER task.
    """
    logits_processors = list()

    # Add JSON generation constraint if we are benchmarking the NER task
    schema = get_ner_schema(ner_tag_names=ner_tag_names)
    logits_processor = JSONLogitsProcessor(
        schema=schema, llm=llm, whitespace_pattern=r" ?"
    )
    logits_processors.append(logits_processor)

    forbidden_token_ids = list()
    forbidden_tokens = ["\n", "\n\n", "\n\n\n", "\t", "\t\t", "\t\t\t"]
    for forbidden_token in forbidden_tokens:
        forbidden_token_ids.extend(
            list(
                llm.get_tokenizer()(forbidden_token, add_special_tokens=False).input_ids
            )
        )
    forbidden_token_ids = list(set(forbidden_token_ids))

    def no_tabs_or_newlines(_: list[int], scores: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(scores)
        for forbidden_token_id in forbidden_token_ids:
            mask[forbidden_token_id] = -math.inf
        return scores + mask

    logits_processors.append(no_tabs_or_newlines)

    return logits_processors
