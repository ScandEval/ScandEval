"""Utility functions related to structured generation."""

import collections.abc as c
import importlib.util
import typing as t

import torch
from pydantic import BaseModel, conlist, create_model

from scandeval.exceptions import NeedsExtraInstalled

if importlib.util.find_spec("xgrammar") is not None:
    import xgrammar as xgr

if t.TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


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
    ner_tag_names: list[str], tokenizer: "PreTrainedTokenizer"
) -> list[c.Callable[[list[int], torch.Tensor], torch.Tensor]]:
    """Get the logits processors for the NER task, used in vLLM.

    Args:
        ner_tag_names:
            The NER tag names.
        tokenizer:
            The tokenizer.

    Returns:
        The logit processors for the NER task.
    """
    if importlib.util.find_spec("xgrammar") is None:
        raise NeedsExtraInstalled(extra="generative")

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer=tokenizer, vocab_size=len(tokenizer)
    )
    compiler = xgr.GrammarCompiler(tokenizer_info=tokenizer_info)
    ner_schema = get_ner_schema(ner_tag_names=ner_tag_names)
    compiled_grammar = compiler.compile_json_schema(schema=ner_schema)
    logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar=compiled_grammar)
    return [logits_processor]
