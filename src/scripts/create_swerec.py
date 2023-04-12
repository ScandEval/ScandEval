"""Create the SweReC-mini sentiment dataset and upload it to the HF Hub."""

import io

import pandas as pd
import requests
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the SweReC-mini sentiment dataset and upload it to the HF Hub."""

    # Define the base download URL
    url = (
        "https://github.com/stoffesvensson/DeepLearning-ThesisWork-Convolutional/"
        "raw/master/webscraper/datasetSameAmountPosNeg"
    )

    # Download the dataset
    csv_str = requests.get(url).text.replace("\r", "")
    csv_file = io.StringIO(csv_str)

    # Convert the dataset to a dataframe
    df = pd.read_csv(csv_file, sep=",", usecols=["text", "rating"])
    df.columns = ["text", "label"]

    # Strip trailing whitespace
    df.text = df.text.str.strip()

    # Capitalize the first letter of each sentence
    df.text = df.text.str.capitalize()

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Create new splits
    full_train_df, val_test_df = train_test_split(
        df, test_size=2048 + 256, random_state=4242, stratify=df.label
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=2048, random_state=4242, stratify=val_test_df.label
    )
    _, train_df = train_test_split(
        full_train_df, test_size=1024, random_state=4242, stratify=full_train_df.label
    )

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split="train"),
        val=Dataset.from_pandas(val_df, split="val"),
        test=Dataset.from_pandas(test_df, split="test"),
    )

    # Create dataset ID
    dataset_id = "ScandEval/swerec-mini"

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
