"""Unit tests for the `question_answering` module."""

import pytest
from contextlib import nullcontext as does_not_raise

from transformers import AutoTokenizer
from scandeval.dataset_configs import (
    SCANDIQA_DA_CONFIG,
    SCANDIQA_NO_CONFIG,
    SCANDIQA_SV_CONFIG,
)
from scandeval.question_answering import QuestionAnswering, prepare_train_examples


@pytest.mark.parametrize(
    argnames=["dataset", "correct_scores"],
    argvalues=[
        (SCANDIQA_DA_CONFIG, (0.24, 4.25)),
        (SCANDIQA_NO_CONFIG, (0.00, 3.72)),
        (SCANDIQA_SV_CONFIG, (0.00, 3.72)),
    ],
    ids=[
        "scandiqa-da",
        "scandiqa-no",
        "scandiqa-sv",
    ],
    scope="class",
)
class TestScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, model_id, dataset):
        benchmark = QuestionAnswering(
            dataset_config=dataset,
            benchmark_config=benchmark_config,
        )
        yield benchmark.benchmark(model_id)[0]["total"]

    def test_em_is_correct(self, scores, correct_scores):
        min_score = scores["test_em"] - scores["test_em_se"]
        max_score = scores["test_em"] + scores["test_em_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_f1"] - scores["test_f1_se"]
        max_score = scores["test_f1"] + scores["test_f1_se"]
        assert min_score <= correct_scores[1] <= max_score


@pytest.mark.parametrize(
    argnames="tokenizer_model_id",
    argvalues=[
        "jonfd/electra-small-nordic",
        "flax-community/swe-roberta-wiki-oscar",
    ],
)
@pytest.mark.parametrize(
    argnames="examples",
    argvalues=[
        dict(
            question=["Hvad er hovedstaden i Sverige?"],
            context=["Sveriges hovedstad er Stockholm."],
            answers=[dict(text=["Stockholm"], answer_start=[22])],
        ),
        dict(
            question=["Hvad er hovedstaden i Sverige?"],
            context=["Sveriges hovedstad er Stockholm." * 100],
            answers=[dict(text=["Sverige"], answer_start=[0])],
        ),
        dict(
            question=["Hvad er hovedstaden i Danmark?"],
            context=["Danmarks hovedstad er København. " * 100],
            answers=[dict(text=["Da"], answer_start=[0])],
        ),
    ],
)
def test_prepare_train_examples(examples, tokenizer_model_id):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
    if (
        hasattr(tokenizer, "model_max_length")
        and tokenizer.model_max_length > 100_000_000
    ):
        tokenizer.model_max_length = 512
    with does_not_raise():
        prepare_train_examples(examples=examples, tokenizer=tokenizer)
