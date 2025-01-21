"""Create the Personal Sum summarisation dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError


def main():
    """Create the Personal Sum summarisation dataset and upload to HF Hub."""
    dataset_url = "https://raw.githubusercontent.com/SmartmediaAI/PersonalSum/refs/heads/main/dataset/PersonalSum_original.csv"
    df = pd.read_csv(dataset_url)

    # Only keep the columns we need and rename them.
    df = df[["Article", "Worker_summary"]]
    df = df.rename(columns={"Article": "text", "Worker_summary": "target_text"})

    # Sort based on text, such that we can avoid having the same article in multiple splits
    df = df.sort_values("text")

    # Strip leading and trailing whitespace
    df["text"] = df["text"].str.strip()
    df["target_text"] = df["target_text"].str.strip()

    # Remove "Oppsummering: " from the start of the target_text
    df["target_text"] = df["target_text"].str.replace(
        r"^Oppsummering: ", "", regex=True
    )

    # Add length columns
    df["text_len"] = df["text"].str.len()
    df["summary_len"] = df["target_text"].str.len()

    # Check bounds
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE

    df = df[df["text_len"].between(lower_bound, upper_bound)]

    # Make splits
    val_size = 128
    test_size = 512

    val_df = df.iloc[:val_size]
    test_df = df.iloc[val_size : val_size + test_size]
    assert (
        val_df.iloc[-1]["text"] != test_df.iloc[0]["text"]
    ), "The last article in the validation set is the same as the first article in the test set. There should be no overlap between the splits."
    train_df = df.iloc[val_size + test_size :]
    assert (
        train_df.iloc[-1]["text"] != test_df.iloc[0]["text"]
    ), "The last article in the training set is the same as the first article in the test set. There should be no overlap between the splits."

    assert len(test_df) > 450, "The test set should have at least 450 samples."

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "ScandEval/personal-sum"

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
