"""Utility functions related to structured generation."""

import importlib.util
import json
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Type

import torch
from pydantic import BaseModel, conlist, create_model
from transformers import SPIECE_UNDERLINE, Pipeline, PreTrainedTokenizerBase

if importlib.util.find_spec("outlines") is not None:
    # from outlines.integrations.transformers import JSONPrefixAllowedTokens
    from outlines.fsm.fsm import FSMState, RegexFSM
    from outlines.fsm.json_schema import build_regex_from_schema
    from outlines.serve.vllm import JSONLogitsProcessor

if TYPE_CHECKING:
    # from outlines.integrations.transformers import JSONPrefixAllowedTokens
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
    ner_tag_names: list[str], tokenizer: PreTrainedTokenizerBase
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


def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Adapt a tokenizer to use to compile the FSM.

    The API of Outlines tokenizers is slightly different to that of `transformers`. In
    addition we need to handle the missing spaces to Llama's tokenizer to be able to
    compile FSMs for this model.

    Args:
        tokenizer:
            The tokenizer of the model.

    Returns:
        The adapted tokenizer.
    """
    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token: str) -> str:
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string

    return tokenizer


class RegexPrefixAllowedTokens:
    """Bias transformers generation based on a regular expression.

    Attributes:
        fsm:
            The finite state machine which is used to bias the logits.
    """

    def __init__(
        self, regex_string: str, tokenizer_or_pipe: PreTrainedTokenizerBase | Pipeline
    ):
        """Compile the FSM that drives the regex-structured generation.

        Args:
            regex_string:
                A string that represents a regular expression.
            tokenizer_or_pipe:
                The tokenizer of the model, or the pipeline object.

        Raises:
            ValueError:
                If the `tokenizer_or_pipe` parameter is not a tokenizer or a pipeline.
        """
        if isinstance(tokenizer_or_pipe, Pipeline):
            tokenizer = tokenizer_or_pipe.tokenizer
        elif isinstance(tokenizer_or_pipe, PreTrainedTokenizerBase):
            tokenizer = tokenizer_or_pipe
        else:
            raise ValueError(
                "The tokenizer_or_pipe parameter must be a tokenizer or a pipeline."
            )
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        self.fsm = RegexFSM(regex_string=regex_string, tokenizer=tokenizer)
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)

        # The generated text with `transformers` include the input token IDs as well,
        # so we use this attribute to keep track of the input token IDs. This allows us
        # to reset the FSM state when the input token IDs change, as well as to only
        # apply the FSM to the generated tokens.
        self._prefix = [-1]

    def __call__(self, batch_id: int, sent: torch.Tensor) -> list[int]:
        """Use the FSM to bias the logits before sampling the next token.

        Args:
            batch_id:
                The index of the current batch.
            sent:
                The tokens of the current sentence.

        Returns:
            The indices of the tokens that are allowed to be sampled next.
        """
        input_ids = sent.tolist()

        # If the prefix token IDs have changed we assume that we are dealing with a new
        # sample and reset the FSM state
        if input_ids[: len(self._prefix)] != self._prefix:
            self._fsm_state = defaultdict(int)
            self._prefix = input_ids
            seq_id = hash(tuple([]))

        else:
            # Remove the prefix token IDs from the input token IDs, as the FSM should
            # only be applied to the generated tokens
            input_ids = input_ids[len(self._prefix) :]

            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            seq_id = hash(tuple(input_ids))
            self._fsm_state[seq_id] = self.fsm.next_state(
                state=FSMState(self._fsm_state[last_seq_id]), token_id=last_token
            )

        allowed_tokens = self.fsm.allowed_token_ids(
            state=FSMState(self._fsm_state[seq_id])
        )
        return allowed_tokens


class JSONPrefixAllowedTokens(RegexPrefixAllowedTokens):
    """Bias transformers generation based on a JSON schema.

    Attributes:
        fsm:
            The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        schema: dict | Type[BaseModel] | str,
        tokenizer_or_pipe: PreTrainedTokenizerBase | Pipeline,
        whitespace_pattern: str | None = None,
    ):
        r"""Compile the FSM that drives the JSON-guided generation.

        Args:
            schema:
                A schema that encodes the structure we want the model to generate.
            tokenizer_or_pipe:
                The tokenizer of the model, or the pipeline object.
            whitespace_pattern:
                Pattern to use for JSON syntactic whitespace (doesn't impact string
                literals). For example, to allow only a single space or newline with
                `whitespace_pattern=r"[\n ]?"`
        """
        if isinstance(schema, dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        elif issubclass(schema, BaseModel):
            schema_str = json.dumps(schema.model_json_schema())
        else:
            raise ValueError(
                f"Cannot parse schema {schema}. The schema must be either "
                + "a Pydantic class, a dictionary or a string that contains the JSON "
                + "schema specification"
            )

        regex_string = build_regex_from_schema(
            schema=schema_str, whitespace_pattern=whitespace_pattern
        )
        super().__init__(regex_string=regex_string, tokenizer_or_pipe=tokenizer_or_pipe)
