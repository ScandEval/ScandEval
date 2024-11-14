"""Unit tests for the `structured_generation_utils` module."""

import json
from typing import Generator

import pytest
from outlines.processors.structured import JSONLogitsProcessor
from scandeval.generation import StopWordCriteria
from scandeval.structured_generation_utils import get_ner_logits_processors
from scandeval.utils import create_model_cache_dir
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteriaList,
)


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
def ner_logits_processors(
    tokenizer,
) -> Generator[list["JSONLogitsProcessor"], None, None]:
    """Logits processors for named entity recognition."""
    yield get_ner_logits_processors(
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
def test_logits_processors(tokenizer, model, ner_logits_processors, input_str):
    """Test the `logits_processors` for named entity recognition."""
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    stopping_criteria = StopWordCriteria(
        stop_word_id_lists=[tokenizer.encode("}"), [tokenizer.eos_token_id]]
    )
    completion = model.generate(
        input_ids,
        generation_config=GenerationConfig(do_sample=False, max_new_tokens=32),
        stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        logits_processor=ner_logits_processors,
    )[0][input_ids.shape[-1] :]
    completion_str = tokenizer.decode(completion, skip_special_tokens=True)
    parsed_output = json.loads(completion_str)
    assert isinstance(parsed_output, dict)
    assert set(list(parsed_output.keys())) == {"person", "location"}
