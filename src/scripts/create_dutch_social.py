"""Create the DutchSocial-mini sentiment dataset and upload it to the HF Hub."""

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
import pandas as pd
from requests import HTTPError


def main() -> None:
    """Create the DutchSocial-mini sentiment dataset and upload it to the HF Hub."""

    # Define the base download URL
    repo_id = "dutch_social"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Create validation split
    val_size = 256
    val_df = val_df.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 1024
    test_df = test_df.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=train_size, random_state=4242)

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
    dataset_id = "ScandEval/dutch-social-mini"

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
