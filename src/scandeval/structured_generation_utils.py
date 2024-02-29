"""Utility functions related to structured generation."""

import importlib.util
import json
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, DefaultDict

import torch
from pydantic import BaseModel, conlist, create_model
from transformers import SPIECE_UNDERLINE, PreTrainedTokenizerBase

from .exceptions import NeedsExtraInstalled

if importlib.util.find_spec("outlines") is not None:
    from outlines.fsm.fsm import RegexFSM
    from outlines.fsm.json_schema import build_regex_from_schema
    from outlines.serve.vllm import JSONLogitsProcessor

if TYPE_CHECKING:
    from outlines.serve.vllm import LLM


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


class JSONPrefixAllowedTokens:
    """Class to be used in `PreTrainedModel.generate` to allow tokens based on JSON."""

    def __init__(
        self,
        ner_tag_names: list[str],
        tokenizer: PreTrainedTokenizerBase,
        forbidden_token_ids: list[int] = list(),
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Args:
            ner_tag_names:
                The NER tag names.
            tokenizer:
                The tokenizer to use for tokenizing the JSON Schema.
            forbidden_token_ids:
                The token IDs that are forbidden to be generated.
        """
        if importlib.util.find_spec("outlines") is None:
            raise NeedsExtraInstalled(extra="generative")

        schema = get_ner_schema(ner_tag_names=ner_tag_names)
        schema_str = json.dumps(schema.model_json_schema())
        regex_string = build_regex_from_schema(
            schema=schema_str, whitespace_pattern=r" ?"
        )
        tokenizer = self.adapt_tokenizer(tokenizer=tokenizer)
        fsm = RegexFSM(regex_string, tokenizer)
        self.forbidden_token_ids = forbidden_token_ids
        self.fsm = fsm

    def __call__(self, batch_id: int, sent: torch.Tensor) -> list[int]:
        """Use the FSM to get the allowed tokens for the next token.

        Args:
            batch_id:
                The batch ID.
            sent:
                The input tensor.

        Returns:
            The allowed tokens.
        """
        sent = sent.tolist()
        seq_id = hash(tuple(sent))

        if len(sent) == 0:
            self.fsm_state: DefaultDict[int, int] = defaultdict(int)
        else:
            last_token = sent[-1]
            last_seq_id = hash(tuple(sent[:-1]))
            self.fsm_state[seq_id] = self.fsm.next_state(
                self.fsm_state[last_seq_id], last_token
            )

        allowed_tokens = [
            token_id
            for token_id in self.fsm.allowed_token_ids(self.fsm_state[seq_id])
            if token_id not in self.forbidden_token_ids
        ]
        return allowed_tokens

    @staticmethod
    def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
        """Adapt vLLM's tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.

        Args:
            tokenizer:
                The tokenizer to adapt.

        Returns:
            The adapted tokenizer.
        """
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            """Patched way to convert a token to a string.

            Args:
                token:
                    The token to convert.

            Returns:
                The string representation of the token.
            """
            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string
        return tokenizer


def get_ner_prefix_allowed_tokens_fn(
    ner_tag_names: list[str], tokenizer: PreTrainedTokenizerBase
) -> JSONPrefixAllowedTokens:
    """Get the prefix allowed tokens function for the NER task, used in `transformers`.

    Args:
        ner_tag_names:
            The NER tag names.
        tokenizer:
            The tokenizer to use for tokenizing the JSON Schema.

    Returns:
        The prefix allowed tokens function for the NER task.
    """
    forbidden_token_ids = list()
    forbidden_tokens = ["\n", "\n\n", "\n\n\n", "\t", "\t\t", "\t\t\t"]
    for forbidden_token in forbidden_tokens:
        forbidden_token_ids.extend(
            list(tokenizer(forbidden_token, add_special_tokens=False).input_ids)
        )
    forbidden_token_ids = list(set(forbidden_token_ids))
    return JSONPrefixAllowedTokens(
        ner_tag_names=ner_tag_names,
        tokenizer=tokenizer,
        forbidden_token_ids=forbidden_token_ids,
    )


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
