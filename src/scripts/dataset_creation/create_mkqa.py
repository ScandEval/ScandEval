# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the MKQA multiple choice datasets and upload them to the HF Hub.

MKQA (https://github.com/apple/ml-mkqa) has the same 10,000 open-domain questions,
taken from Natural Questions, translated into 26 languages. Each question comes with a
short answer but no context, so we turn the questions into multiple choice knowledge
questions: the correct answer plus three answers to other questions of the same answer
type.
"""

import gzip
import io
import json
import random
import typing as t
import unicodedata

import requests
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_OPTION,
    MIN_NUM_CHARS_IN_QUESTION,
)
from dataset_utils import MAX_TEST, MAX_TRAIN, MAX_VAL, SEED, upload_private
from datasets import Dataset, DatasetDict

SOURCE_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"

# The EuroEval languages that MKQA has translations for
LANGUAGES = [
    "da",
    "de",
    "en",
    "es",
    "fi",
    "fr",
    "hu",
    "it",
    "nl",
    "no",
    "pl",
    "pt",
    "sv",
]

# Answer types that make sense as multiple choice options. Binary answers have too few
# distinct options, long answers are sentences, and unanswerable questions have no
# answer.
ANSWER_TYPES = ["entity", "date", "number", "number_with_unit", "short_phrase"]

LETTERS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the MKQA multiple choice datasets and upload them to the HF Hub."""
    rows = load_rows()
    for language in LANGUAGES:
        dataset = create_dataset(rows=rows, language=language)
        upload_private(dataset=dataset, repository_id=f"EuroEval/mkqa-{language}-mini")


def _normalise(text: str) -> str:
    """Normalise an answer, so that answers can be compared.

    Args:
        text:
            The answer or alias.

    Returns:
        The text with Unicode compatibility forms folded, whitespace collapsed and
        case removed.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


class Sample(t.TypedDict):
    """A usable MKQA question and its answer in one language."""

    example_id: str
    question: str
    answer_type: str
    answer: str
    accepted: set[str]


def _build_split(samples: list[Sample], pool: list[Sample], language: str) -> Dataset:
    """Build one split, with distractors taken from the same split.

    Args:
        samples:
            The usable questions and answers of the split.
        pool:
            The samples that the distractors are drawn from.
        language:
            The MKQA language code.

    Returns:
        A dataset with the columns "text" and "label".
    """
    pool_by_type: dict[str, list[str]] = {}
    for sample in pool:
        pool_by_type.setdefault(sample["answer_type"], []).append(sample["answer"])

    texts: list[str] = []
    labels: list[str] = []
    for sample in samples:
        rng = random.Random(f"{SEED}-{language}-{sample['example_id']}")
        candidates = sorted(
            {
                _normalise(text=answer): answer
                for answer in pool_by_type[sample["answer_type"]]
                if _normalise(text=answer) not in sample["accepted"]
            }.values()
        )
        if len(candidates) < len(LETTERS) - 1:
            continue
        options = [sample["answer"], *rng.sample(candidates, k=len(LETTERS) - 1)]
        rng.shuffle(options)
        lines = [f"{letter}. {option}" for letter, option in zip(LETTERS, options)]
        texts.append(
            f"{sample['question']}\n{CHOICES_MAPPING[language]}:\n" + "\n".join(lines)
        )
        labels.append(LETTERS[options.index(sample["answer"])])
    return Dataset.from_dict(mapping=dict(text=texts, label=labels))


class Answer(t.TypedDict):
    """An answer to an MKQA question in one language."""

    type: str
    text: str | None
    aliases: list[str] | None


class MkqaRow(t.TypedDict):
    """An MKQA question with its translations and answers in all languages."""

    example_id: int | str
    queries: dict[str, str]
    answers: dict[str, list[Answer]]


def create_dataset(
    rows: list[MkqaRow],
    language: str,
    train_size: int = MAX_TRAIN,
    val_size: int = MAX_VAL,
    test_size: int = MAX_TEST,
) -> DatasetDict:
    """Create the multiple choice dataset for one language.

    Args:
        rows:
            The MKQA rows, with the queries and answers for all languages.
        language:
            The MKQA language code, such as "da".
        train_size:
            The maximum number of training samples.
        val_size:
            The maximum number of validation samples.
        test_size:
            The maximum number of test samples.

    Returns:
        A dataset dictionary with the splits "train", "val" and "test", each with the
        columns "text" and "label".
    """
    shuffled_rows = sorted(rows, key=lambda row: str(row["example_id"]))
    random.Random(SEED).shuffle(shuffled_rows)

    total_size = train_size + val_size + test_size
    test_end = round(len(shuffled_rows) * test_size / total_size)
    val_end = test_end + round(len(shuffled_rows) * val_size / total_size)
    rows_by_split = dict(
        train=shuffled_rows[val_end:],
        val=shuffled_rows[test_end:val_end],
        test=shuffled_rows[:test_end],
    )
    max_size_by_split = dict(train=train_size, val=val_size, test=test_size)

    splits: dict[str, Dataset] = dict()
    for split, split_rows in rows_by_split.items():
        samples = [
            sample
            for sample in (
                _get_sample(row=row, language=language) for row in split_rows
            )
            if sample is not None
        ][: max_size_by_split[split]]
        splits[split] = _build_split(samples=samples, pool=samples, language=language)
    return DatasetDict({split: dataset for split, dataset in splits.items()})


def _get_sample(row: MkqaRow, language: str) -> Sample | None:
    """Get the question and answer of a row, if they are usable.

    Args:
        row:
            An MKQA row.
        language:
            The MKQA language code.

    Returns:
        The example ID, question, answer type, answer and the normalised texts of all
        accepted answers and their aliases, or None if the row cannot be used as a
        multiple choice question. The first answer is the one that is used as the
        correct option.
    """
    question = row["queries"][language].strip()
    answers = row["answers"][language]
    if not answers or answers[0]["type"] not in ANSWER_TYPES:
        return None
    answer = (answers[0]["text"] or "").strip()
    if not MIN_NUM_CHARS_IN_QUESTION <= len(question) <= MAX_NUM_CHARS_IN_QUESTION:
        return None
    if not MIN_NUM_CHARS_IN_OPTION <= len(answer) <= MAX_NUM_CHARS_IN_OPTION:
        return None
    return Sample(
        example_id=str(row["example_id"]),
        question=question,
        answer_type=answers[0]["type"],
        answer=answer,
        accepted={
            _normalise(text=text)
            for accepted_answer in answers
            for text in [
                accepted_answer["text"],
                *(accepted_answer.get("aliases") or []),
            ]
            if isinstance(text, str)
        },
    )


def load_rows() -> list[MkqaRow]:
    """Download the MKQA questions and answers.

    Returns:
        The MKQA rows, with the queries and answers for all languages.
    """
    response = requests.get(url=SOURCE_URL, timeout=300)
    response.raise_for_status()
    with gzip.open(io.BytesIO(response.content), mode="rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    main()
