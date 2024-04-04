"""Create the DanishCitizenTests-mini dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the DanishCitizenTests-mini dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/danish-citizen-tests"

    # Download the dataset
    dataset = load_dataset(path=repo_id, split="train")
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Rename the columns
    df.rename(columns=dict(answer="label", question="instruction"), inplace=True)

    # Make a `text` column with all the options in it
    texts = list()
    for _, row in df.iterrows():
        text = (
            row.instruction.replace("\n", " ").strip() + "\n"
            "Svarmuligheder:\n"
            "a. " + row.option_a.replace("\n", " ").strip() + "\n"
            "b. " + row.option_b.replace("\n", " ").strip()
        )
        if row.option_c is not None:
            text += "\nc. " + row.option_c.replace("\n", " ").strip()
        texts.append(text)
    df["text"] = texts

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    df = df[["text", "label", "test_type"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 128
    traintest_arr, val_arr = train_test_split(
        df, test_size=val_size, random_state=4242, stratify=df.test_type
    )
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create test split
    test_size = 512
    train_arr, test_arr = train_test_split(
        traintest_df,
        test_size=test_size,
        random_state=4242,
        stratify=traintest_df.test_type,
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

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
    dataset_id = "ScandEval/danish-citizen-tests"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
