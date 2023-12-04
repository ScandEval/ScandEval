"""Create the NordjyllandNews-mini summarisation dataset."""

from huggingface_hub import HfApi
import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from requests import HTTPError


def main():
    """Create the NordjyllandNews-mini summarisation dataset and upload to HF Hub."""

    dataset_id = "alexandrainst/nordjylland-news-summarization"

    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = dataset.rename_columns(column_mapping=dict(summary="target_text"))

    train_df = dataset["train"].to_pandas()
    val_df = dataset["val"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Create validation split
    val_size = 256
    val_df = val_df.sample(n=val_size, random_state=4242)
    val_df = val_df.reset_index(drop=True)

    # Create test split
    test_size = 2048
    test_df = test_df.sample(n=test_size, random_state=4242)
    test_df = test_df.reset_index(drop=True)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=train_size, random_state=4242)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    mini_dataset_id = "ScandEval/nordjylland-news-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id)


if __name__ == "__main__":
    main()
