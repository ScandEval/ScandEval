"""Create the FoQA dataset and upload them to the HF Hub."""

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the FoQA datasets and upload them to the HF Hub."""
    dataset_id = "alexandrainst/foqa"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["val"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Only work with samples where the context is not very large or small
    train_lengths = train_df.context.str.len()
    val_lengths = val_df.context.str.len()
    test_lengths = test_df.context.str.len()
    train_df = train_df[
        train_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]
    val_df = val_df[
        val_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]
    test_df = test_df[
        test_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]

    # Only work with samples where the question is not very large or small
    train_question_lengths = train_df.question.str.len()
    val_question_lengths = val_df.question.str.len()
    test_question_lengths = test_df.question.str.len()
    train_df = train_df[
        train_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]
    val_df = val_df[
        val_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]
    test_df = test_df[
        test_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Extract information on which examples contain an answer
    def has_answer(example: dict) -> bool:
        return len(example["text"]) > 0 and example["text"][0] != ""

    train_has_answer: pd.Series = train_df.answers.map(has_answer)
    val_has_answer: pd.Series = val_df.answers.map(has_answer)
    test_has_answer: pd.Series = test_df.answers.map(has_answer)

    # Only work with the questions having answers in the context
    train_df = train_df.loc[train_has_answer]
    val_df = val_df.loc[val_has_answer]
    test_df = test_df.loc[test_has_answer]

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    mini_dataset_id = "ScandEval/foqa"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id, private=True)


if __name__ == "__main__":
    main()
