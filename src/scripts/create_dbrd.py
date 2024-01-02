"""Create the DBRD-mini SENT dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def main():
    """Create the DBRD-mini SENT dataset and uploads it to the HF Hub."""

    # Define dataset ID
    repo_id = "dbrd"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Strip trailing whitespace
    train_df.text = train_df.text.str.strip()
    test_df.text = test_df.text.str.strip()

    # Capitalize the first letter of each sentence
    train_df.text = train_df.text.str.capitalize()
    test_df.text = test_df.text.str.capitalize()

    # Remove duplicates
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    # Create the label column
    label_mapping = {0: "negative", 1: "positive"}
    train_df["label"] = train_df["label"].map(label_mapping)
    test_df["label"] = test_df["label"].map(label_mapping)

    full_train_df, val_df = train_test_split(
        train_df, test_size=256, random_state=703, stratify=train_df.label
    )
    _, train_df = train_test_split(
        full_train_df, test_size=1024, random_state=703, stratify=full_train_df.label
    )

    _, test_df = train_test_split(
        test_df, test_size=2048, random_state=703, stratify=test_df.label
    )

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "ScandEval/dbrd-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id)


if __name__ == "__main__":
    main()
