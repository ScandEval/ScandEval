"""Public-interface tests for creating the MKQA multiple choice datasets."""

import typing as t

import pytest
from datasets import DatasetDict

from scripts.dataset_creation import create_mkqa


class TestCreateDataset:
    """Tests for the `create_dataset` function."""

    def test_creation_is_deterministic(
        self, rows: list[dict[str, t.Any]], dataset: DatasetDict
    ) -> None:
        """Creating the dataset twice gives the same samples."""
        again = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=10, val_size=5, test_size=20
        )
        assert again["test"].to_list() == dataset["test"].to_list()

    def test_distractors_come_from_the_same_split(self, dataset: DatasetDict) -> None:
        """The options of a sample are answers to questions in the same split."""
        for split in dataset.values():
            answers = {
                f"svar {sample['text'].split(chr(10))[0].split()[2].rstrip('?')}"
                for sample in split
            }
            for sample in split:
                options = {option[3:] for option in sample["text"].split("\n")[2:]}
                assert options <= answers

    def test_four_distinct_options(self, dataset: DatasetDict) -> None:
        """Every sample has the options a to d, and they are all different."""
        for split in dataset.values():
            for sample in split:
                options = sample["text"].split("\n")[2:]
                assert [option[:2] for option in options] == ["a.", "b.", "c.", "d."]
                assert len({option[3:] for option in options}) == 4

    def test_label_points_to_the_answer(self, dataset: DatasetDict) -> None:
        """The label is the letter of the option holding the answer."""
        for split in dataset.values():
            for sample in split:
                question, header, *options = sample["text"].split("\n")
                assert header == "Svarmuligheder:"
                number = question.removeprefix("Spørgsmål nummer ").removesuffix("?")
                by_letter = {option[0]: option[3:] for option in options}
                assert by_letter[sample["label"]] == f"svar {number}"

    def test_same_questions_in_the_same_split_in_all_languages(
        self, rows: list[dict[str, t.Any]], dataset: DatasetDict
    ) -> None:
        """A question is in the same split in Danish and English."""
        english = create_mkqa.create_dataset(
            rows=rows, language="en", train_size=10, val_size=5, test_size=20
        )
        for split in ("train", "val", "test"):
            danish_numbers = [
                sample["text"].split("\n")[0].split()[2].rstrip("?")
                for sample in dataset[split]
            ]
            english_numbers = [
                sample["text"].split("\n")[0].split()[2].rstrip("?")
                for sample in english[split]
            ]
            assert danish_numbers == english_numbers

    def test_split_does_not_depend_on_language_specific_filtering(self) -> None:
        """A question rejected in one language does not shift splits in the others."""
        rows = [
            _row(index=index, english_question="Short?" if index % 5 == 0 else None)
            for index in range(100)
        ]
        danish = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=10, val_size=5, test_size=20
        )
        english = create_mkqa.create_dataset(
            rows=rows, language="en", train_size=10, val_size=5, test_size=20
        )

        danish_splits = _split_by_question_number(dataset=danish)
        english_splits = _split_by_question_number(dataset=english)

        shared = danish_splits.keys() & english_splits.keys()
        assert {danish_splits[number] for number in shared} == {"train", "val", "test"}
        for number in shared:
            assert danish_splits[number] == english_splits[number]
        assert not any(int(number) % 5 == 0 for number in english_splits)

    def test_splits_and_columns(self, dataset: DatasetDict) -> None:
        """The dataset has train, val and test splits with text and label."""
        assert set(dataset.keys()) == {"train", "val", "test"}
        assert [dataset[split].num_rows for split in ("train", "val", "test")] == [
            10,
            5,
            20,
        ]
        for split in dataset.values():
            assert split.column_names == ["text", "label"]

    def test_splits_do_not_share_questions(self, dataset: DatasetDict) -> None:
        """No question is in more than one split."""
        questions = [
            sample["text"].split("\n")[0]
            for split in dataset.values()
            for sample in split
        ]
        assert len(questions) == len(set(questions))

    def test_unusable_answer_types_are_left_out(self, dataset: DatasetDict) -> None:
        """Unanswerable, binary and long answers are not used."""
        questions = [
            sample["text"].split("\n")[0]
            for split in dataset.values()
            for sample in split
        ]
        for index in (100, 101, 102):
            assert f"Spørgsmål nummer {index}?" not in questions


def _row(
    index: int, answer_type: str = "entity", english_question: str | None = None
) -> dict[str, t.Any]:
    """Create an MKQA row with a Danish and an English question.

    Args:
        index:
            The index of the row, which makes the question and answer unique.
        answer_type:
            The MKQA answer type of the answer.
        english_question:
            The English question, or None to use a question that is unique to the row.

    Returns:
        An MKQA row.
    """
    return dict(
        example_id=index,
        query=f"question {index}",
        queries=dict(
            da=f"Spørgsmål nummer {index}?",
            en=english_question or f"Question number {index}?",
        ),
        answers=dict(
            da=[dict(type=answer_type, text=f"svar {index}", aliases=[])],
            en=[dict(type=answer_type, text=f"answer {index}", aliases=[])],
        ),
    )


def _split_by_question_number(dataset: DatasetDict) -> dict[str, str]:
    """Map the number in each question to the split that the question is in.

    Args:
        dataset:
            A dataset created from rows made with `_row`.

    Returns:
        A dictionary mapping question numbers to split names.
    """
    return {
        sample["text"].split("\n")[0].split()[2].rstrip("?"): str(split)
        for split, samples in dataset.items()
        for sample in samples
    }


@pytest.fixture(scope="module")
def dataset(rows: list[dict[str, t.Any]]) -> DatasetDict:
    """Create a small Danish dataset.

    Args:
        rows:
            MKQA rows.

    Returns:
        The Danish dataset with 10 / 5 / 20 samples.
    """
    return create_mkqa.create_dataset(
        rows=rows, language="da", train_size=10, val_size=5, test_size=20
    )


@pytest.fixture(scope="module")
def rows() -> list[dict[str, t.Any]]:
    """Provide 60 usable rows plus rows that cannot be used.

    Returns:
        MKQA rows.
    """
    unusable = [
        _row(index=100, answer_type="unanswerable"),
        _row(index=101, answer_type="binary"),
        _row(index=102, answer_type="long_answer"),
    ]
    return [_row(index=index) for index in range(60)] + unusable
