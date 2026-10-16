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


def create_dataset(
    rows: list[dict[str, t.Any]],
    language: str,
    train_size: int = MAX_TRAIN,
    val_size: int = MAX_VAL,
    test_size: int = MAX_TEST,
) -> DatasetDict:
    """Create the multiple choice dataset for one language.

    MKQA has no splits, so the questions are shuffled with the same seed for every
    language and then divided into test, validation and training splits. A question
    therefore ends up in the same split in all languages that keep it.

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

    samples = [
        sample
        for sample in (_get_sample(row=row, language=language) for row in shuffled_rows)
        if sample is not None
    ]
    test_samples = samples[:test_size]
    val_samples = samples[test_size : test_size + val_size]
    train_samples = samples[test_size + val_size : test_size + val_size + train_size]

    return DatasetDict(
        {
            "train": _build_split(
                samples=train_samples, pool=train_samples, language=language
            ),
            "val": _build_split(
                samples=val_samples, pool=val_samples, language=language
            ),
            "test": _build_split(
                samples=test_samples, pool=test_samples, language=language
            ),
        }
    )


def _build_split(
    samples: list[dict[str, t.Any]], pool: list[dict[str, t.Any]], language: str
) -> Dataset:
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
        pool_by_type.setdefault(sample["type"], []).append(sample["answer"])

    texts: list[str] = []
    labels: list[str] = []
    for sample in samples:
        rng = random.Random(f"{SEED}-{language}-{sample['example_id']}")
        forbidden = {sample["answer"].casefold(), *sample["aliases"]}
        candidates = sorted(
            {
                answer.casefold(): answer
                for answer in pool_by_type[sample["type"]]
                if answer.casefold() not in forbidden
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


def _get_sample(row: dict[str, t.Any], language: str) -> dict[str, t.Any] | None:
    """Get the question and answer of a row, if they are usable.

    Args:
        row:
            An MKQA row.
        language:
            The MKQA language code.

    Returns:
        The example ID, question, answer type, answer and aliases of the answer, or
        None if the row cannot be used as a multiple choice question.
    """
    question = row["queries"][language].strip()
    answers = row["answers"][language]
    if not answers or answers[0]["type"] not in ANSWER_TYPES:
        return None
    answer = answers[0]["text"].strip()
    if not MIN_NUM_CHARS_IN_QUESTION <= len(question) <= MAX_NUM_CHARS_IN_QUESTION:
        return None
    if not MIN_NUM_CHARS_IN_OPTION <= len(answer) <= MAX_NUM_CHARS_IN_OPTION:
        return None
    return dict(
        example_id=str(row["example_id"]),
        question=question,
        type=answers[0]["type"],
        answer=answer,
        aliases={alias.strip().casefold() for alias in answers[0].get("aliases") or []},
    )


def load_rows() -> list[dict[str, t.Any]]:
    """Download the MKQA questions and answers.

    Returns:
        The 10,000 MKQA rows, with the queries and answers for all languages.
    """
    response = requests.get(url=SOURCE_URL, timeout=300)
    response.raise_for_status()
    with gzip.open(io.BytesIO(response.content), mode="rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    main()
