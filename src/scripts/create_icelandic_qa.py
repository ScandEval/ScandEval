"""Create the ScandiQA-mini datasets and upload them to the HF Hub."""

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
    """Create the ScandiQA-mini datasets and upload them to the HF Hub."""
    repo_id = "mideind/icelandic_qa_scandeval"

    dataset = load_dataset(path=repo_id, token=True, split="train")

    df = dataset.to_pandas().rename(columns={"example_id": "id"})

    # Only work with samples where the context is not very large or small
    lengths = df.context.str.len()
    df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

    # Only work with samples where the question is not very large or small
    lengths = df.question.str.len()
    df_with_no_outliers = df[
        lengths.between(MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION)
    ]

    # Only work with the questions having answers in the context
    has_answer: pd.Series = df_with_no_outliers.apply(
        lambda row: row["answer"].lower() in row["context"].lower(), axis=1
    )

    df_with_answer: pd.DataFrame = df_with_no_outliers.loc[has_answer]

    # Make the `answers` column a dictionary with the answer and the answer start
    df_with_answer["answers"] = df_with_answer.apply(
        lambda row: {
            "text": [row["answer"]],
            "answer_start": row["context"].lower().index(row["answer"].lower()),
        },
        axis=1,
    )

    # Remove the old `answer` column
    df_with_answer = df_with_answer.drop(columns=["answer"])

    # Create validation split
    val_size = 25
    val_df = df_with_answer.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 75
    df_with_answer_filtered: pd.DataFrame = df_with_answer.loc[
        ~df_with_answer.index.isin(val_df.index)
    ]
    test_df = df_with_answer_filtered.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 275
    full_train_df_with_answer = df_with_answer_filtered.loc[
        ~df_with_answer_filtered.index.isin(test_df.index)
    ]
    train_df = full_train_df_with_answer.sample(n=train_size, random_state=4242)

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
    dataset_id = "ScandEval/icelandic-qa"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
