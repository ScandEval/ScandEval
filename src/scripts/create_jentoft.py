"""Create the Jentoft linguistic acceptability dataset."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError


def main():
    """Create the Jentoft linguistic acceptability dataset and upload to HF Hub."""
    dataset_urls = {
        "val": "https://raw.githubusercontent.com/matias-jjj/matias_master/refs/heads/main/norgec_dataset/ask_exp_dev.tsv",
        "test": "https://raw.githubusercontent.com/matias-jjj/matias_master/refs/heads/main/norgec_dataset/ask_exp_test.tsv",
        "train": "https://raw.githubusercontent.com/matias-jjj/matias_master/refs/heads/main/norgec_dataset/ask_exp_train.tsv",
    }

    full_val_df = prepare_dataset(dataset_url=dataset_urls["val"])
    full_test_df = prepare_dataset(dataset_url=dataset_urls["test"])
    full_train_df = prepare_dataset(dataset_url=dataset_urls["train"])

    # Remove overlapping texts between splits
    full_train_df, full_val_df, full_test_df = remove_overlapping_samples(
        full_train_df, full_val_df, full_test_df
    )

    assert isinstance(full_train_df, pd.DataFrame)
    assert isinstance(full_val_df, pd.DataFrame)
    assert isinstance(full_test_df, pd.DataFrame)

    val_size = 256
    test_size = 2048
    train_size = 1024

    def sample_dataset(full_df: pd.DataFrame, size: int) -> pd.DataFrame:
        """Sample a dataset with an equal number of 'correct' and 'incorrect' samples.

        Args:
            full_df:
                The full dataset.
            size:
                The size of the sampled dataset.

        Returns:
            The sampled dataset.
        """
        half_size = size // 2
        assert (
            full_df["label"] == "correct"
        ).sum() >= half_size, "Not enough 'correct' samples"

        correct_df = full_df[full_df["label"] == "correct"].sample(
            n=half_size, random_state=4242
        )
        incorrect_df = full_df[full_df["label"] == "incorrect"].sample(
            n=half_size, random_state=4242
        )
        return pd.concat([correct_df, incorrect_df])

    val_df = sample_dataset(full_df=full_val_df, size=val_size)
    test_df = sample_dataset(full_df=full_test_df, size=test_size)
    train_df = sample_dataset(full_df=full_train_df, size=train_size)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN, preserve_index=False),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION, preserve_index=False),
        test=Dataset.from_pandas(test_df, split=Split.TEST, preserve_index=False),
        full_train=Dataset.from_pandas(
            full_train_df, split=Split.TRAIN, preserve_index=False
        ),
        full_val=Dataset.from_pandas(
            full_val_df, split=Split.VALIDATION, preserve_index=False
        ),
        full_test=Dataset.from_pandas(
            full_test_df, split=Split.TEST, preserve_index=False
        ),
    )

    # Create dataset ID
    dataset_id = "ScandEval/jentoft-la"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def prepare_dataset(dataset_url: str) -> pd.DataFrame:
    """Prepare the dataset from a URL.

    Args:
        dataset_url:
            The URL to the dataset.

    Returns:
        The prepared dataset.
    """
    df = pd.read_csv(dataset_url, sep="\t", on_bad_lines="skip")

    # Make label column
    df["label"] = df["ERROR"].apply(
        lambda x: "correct" if x == "no_error" else "incorrect"
    )

    # Rename the columns
    df = df.rename(columns={"SOURCE": "text"})

    # Only keep relevant columns
    df = df[["text", "label"]]

    # Remove text duplicates
    df = df.drop_duplicates(subset=["text"])

    # Remove samples with with 5 or fewer words
    df = df[df["text"].str.split().apply(len) > 5]
    return df


def remove_overlapping_samples(
    full_train_df: pd.DataFrame, full_val_df: pd.DataFrame, full_test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure that a sample is only present in one split.

    Args:
        full_train_df:
            The full training dataset.
        full_val_df:
            The full validation dataset.
        full_test_df:
            The full test dataset.

    Returns:
        The cleaned training, validation, and test datasets.
    """
    train_texts = set(full_train_df["text"])
    val_texts = set(full_val_df["text"])
    test_texts = set(full_test_df["text"])

    # Find overlaps
    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)

    # Print overlap statistics
    print(f"Overlapping texts between train and validation: {len(train_val_overlap)}")
    print(f"Overlapping texts between train and test: {len(train_test_overlap)}")
    print(f"Overlapping texts between validation and test: {len(val_test_overlap)}")

    # Remove overlapping texts from validation and test sets, prioritize train > validation > test
    clean_val_df = full_val_df[~full_val_df["text"].isin(train_texts)]
    clean_test_df = full_test_df[
        ~full_test_df["text"].isin(train_texts) & ~full_test_df["text"].isin(val_texts)
    ]

    # The training set remains unchanged
    clean_train_df = full_train_df

    # Verify no more overlaps
    assert len(set(clean_train_df["text"]).intersection(set(clean_val_df["text"]))) == 0
    assert (
        len(set(clean_train_df["text"]).intersection(set(clean_test_df["text"]))) == 0
    )
    assert len(set(clean_val_df["text"]).intersection(set(clean_test_df["text"]))) == 0

    # Print final dataset sizes
    print("\nDataset sizes after removing overlapping texts:")
    print(f"Train: {len(clean_train_df)} (originally {len(full_train_df)})")
    print(f"Validation: {len(clean_val_df)} (originally {len(full_val_df)})")
    print(f"Test: {len(clean_test_df)} (originally {len(full_test_df)})")
    return clean_train_df, clean_val_df, clean_test_df


if __name__ == "__main__":
    main()
