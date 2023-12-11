"""Unit tests for the `text_to_text` module."""

import pytest

from scandeval.dataset_configs import (
    MLSUM_CONFIG,
    NO_SAMMENDRAG_CONFIG,
    NORDJYLLAND_NEWS_CONFIG,
    RRN_CONFIG,
    SWEDN_CONFIG,
    WIKI_LINGUA_NL_CONFIG,
)
from scandeval.text_to_text import TextToText


@pytest.mark.parametrize(
    argnames=["dataset", "correct_scores"],
    argvalues=[
        (NORDJYLLAND_NEWS_CONFIG, (-1000, -1000)),
        (SWEDN_CONFIG, (-1000, -1000)),
        (NO_SAMMENDRAG_CONFIG, (-1000, -1000)),
        (RRN_CONFIG, (-1000, -1000)),
        (MLSUM_CONFIG, (-1000, -1000)),
        (WIKI_LINGUA_NL_CONFIG, (-1000, -1000)),
    ],
    ids=[
        "nordjylland-news",
        "swedn",
        "no-sammendrag",
        "rrn",
        "mlsum",
        "wiki-lingua-nl",
    ],
    scope="class",
)
class TestGenerativeScores:
    @pytest.fixture(scope="class")
    def scores(self, benchmark_config, generative_model_id, dataset):
        benchmark = TextToText(
            dataset_config=dataset,
            benchmark_config=benchmark_config,
        )
        yield benchmark.benchmark(generative_model_id)[0]["total"]

    def test_mcc_is_correct(self, scores, correct_scores):
        min_score = scores["test_bertscore"] - scores["test_bertscore_se"]
        max_score = scores["test_bertscore"] + scores["test_bertscore_se"]
        assert min_score <= correct_scores[0] <= max_score

    def test_macro_f1_is_correct(self, scores, correct_scores):
        min_score = scores["test_rouge_l"] - scores["test_rouge_l_se"]
        max_score = scores["test_rouge_l"] + scores["test_rouge_l_se"]
        assert min_score <= correct_scores[1] <= max_score
