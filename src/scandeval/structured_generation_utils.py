"""Utility functions related to structured generation."""

import collections.abc as c
import importlib.util
import typing as t

import torch
from pydantic import BaseModel, conlist, create_model

if importlib.util.find_spec("outlines") is not None:
    from outlines.integrations.vllm import JSONLogitsProcessor

if t.TYPE_CHECKING:
    from vllm import LLM


def get_ner_schema(ner_tag_names: list[str]) -> type[BaseModel]:
    """Get the schema for the NER answer format, used for structured generation.

    Args:
        ner_tag_names:
            The NER tag names.

    Returns:
        The schema for the NER answer format.
    """
    keys_and_their_types: dict[str, t.Any] = {
        tag_name: (conlist(str, max_length=5), ...) for tag_name in ner_tag_names
    }
    schema = create_model("AnswerFormat", **keys_and_their_types)
    return schema


def get_ner_logits_processors(
    ner_tag_names: list[str], llm: "LLM"
) -> list[c.Callable[[list[int], torch.Tensor], torch.Tensor]]:
    """Get the logits processors for the NER task, used in vLLM.

    Args:
        ner_tag_names:
            The NER tag names.
        llm:
            The vLLM model.

    Returns:
        The logit processors for the NER task.
    """
    logits_processor = JSONLogitsProcessor(
        schema=get_ner_schema(ner_tag_names=ner_tag_names),
        llm=llm,
        whitespace_pattern=r" ?",
    )
    return [logits_processor]
