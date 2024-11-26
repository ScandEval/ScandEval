"""Unit tests for the `structured_generation_utils` module."""

import json
from typing import TYPE_CHECKING, Generator

import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteriaList,
)

from scandeval.generation import StopWordCriteria
from scandeval.structured_generation_utils import get_ner_prefix_allowed_tokens_fn
from scandeval.utils import create_model_cache_dir

if TYPE_CHECKING:
    from outlines.integrations.transformers import JSONPrefixAllowedTokens


@pytest.fixture(scope="module")
def tokenizer(generative_model_id) -> Generator[PreTrainedTokenizerBase, None, None]:
    """A tokenizer for a generative model."""
    yield AutoTokenizer.from_pretrained(generative_model_id)


@pytest.fixture(scope="module")
def model(
    generative_model_id, benchmark_config
) -> Generator[PreTrainedModel, None, None]:
    """A generative model."""
    model_cache_dir = create_model_cache_dir(
        cache_dir=benchmark_config.cache_dir, model_id=generative_model_id
    )
    yield AutoModelForCausalLM.from_pretrained(
        generative_model_id, cache_dir=model_cache_dir, torch_dtype=None
    )


@pytest.fixture(scope="module")
def ner_prefix_allowed_tokens_fn(
    tokenizer,
) -> Generator["JSONPrefixAllowedTokens", None, None]:
    """A prefix_allowed_tokens_fn for named entity recognition."""
    yield get_ner_prefix_allowed_tokens_fn(
        ner_tag_names=["person", "location"], tokenizer=tokenizer
    )


@pytest.mark.parametrize(
    argnames="input_str",
    argvalues=[
        "John works at Novo Nordisk.",
        "Peter lives in Copenhagen.",
        "He loves to surf.",
    ],
    ids=["person-no-location", "person-location", "no-person-no-location"],
)
def test_prefix_allowed_tokens_fn(
    tokenizer, model, ner_prefix_allowed_tokens_fn, input_str
):
    """Test the `prefix_allowed_tokens_fn` for named entity recognition."""
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    stopping_criteria = StopWordCriteria(
        stop_word_id_lists=[tokenizer.encode("}"), [tokenizer.eos_token_id]]
    )
    completion = model.generate(
        input_ids,
        generation_config=GenerationConfig(do_sample=False, max_new_tokens=32),
        stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        prefix_allowed_tokens_fn=ner_prefix_allowed_tokens_fn,
    )[0][input_ids.shape[-1] :]
    completion_str = tokenizer.decode(completion, skip_special_tokens=True)
    parsed_output = json.loads(completion_str)
    assert isinstance(parsed_output, dict)
    assert set(list(parsed_output.keys())) == {"person", "location"}
