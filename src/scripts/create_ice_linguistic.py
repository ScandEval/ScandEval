"""Create a linguistic acceptability dataset from the Icelandic Linguistic Benchmarks dataset."""

from ast import literal_eval

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests.exceptions import HTTPError


def main():
    """Create a linguistic acceptability dataset from the Icelandic Linguistic Benchmarks dataset."""
    dataset_url = "https://raw.githubusercontent.com/stofnun-arna-magnussonar/ice_linguistic_benchmarks/refs/heads/main/ice_benchmark_set.jsonl"

    # Download the dataset and convert it to a dataframe
    response = requests.get(dataset_url)
    lines = [literal_eval(line) for line in response.text.split("\n") if line.strip()]
    df = pd.DataFrame(lines)

    train_df = prepare_dataframe(df=df)

    # Validation split
    val_size = 32
    val_df = train_df.sample(n=val_size, random_state=4242)
    train_df = train_df.drop(val_df.index)

    # Test split
    test_size = 256
    test_df = train_df.sample(n=test_size, random_state=4242)
    train_df = train_df.drop(test_df.index)

    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN, preserve_index=False),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION, preserve_index=False),
        test=Dataset.from_pandas(test_df, split=Split.TEST, preserve_index=False),
    )

    # Create dataset ID
    dataset_id = "ScandEval/ice-linguistic"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a dataframe from a dataset.

    Args:
        df:
            The dataframe to prepare

    Returns:
        A dataframe with the prepared dataset.
    """
    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Only extract rows where the task is to determine if a sentence is grammatically correct/incorrect.
    df = df[
        df["input"].str.contains("Is the following Icelandic sentence grammatically")
    ]

    # extract the text between \n\n TEXT \n\n and strip the text
    df["text"] = df["input"].str.extract(r"\n\n(.*)\n\n")
    df["text"] = df["text"].str.strip()

    # Remove samples with too few tokens
    df = df[df["text"].str.split().map(len) > 5]

    def make_label(row: pd.Series) -> str:
        # asking if the sentence is incorrect.
        if "incorrect" in row["input"]:
            return "nei" if row["answer"] == "Yes" else "já"

        # ...asking if the sentence is correct
        return "já" if row["answer"] == "Yes" else "nei"

    df["label"] = df.apply(make_label, axis=1)

    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=4242).reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
