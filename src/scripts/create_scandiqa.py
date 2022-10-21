"""Create the ScandiQA-mini datasets and upload them to the HF Hub."""

import pandas as pd
from datasets.arrow_dataset import Dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the ScandiQA-mini datasets and upload them to the HF Hub."""

    dataset_id = "alexandrainst/scandiqa"

    # Iterate over the Danish, Norwegian and Swedish languages
    for language in ["da", "no", "sv"]:

        # Load the datasets from the `alexandrainst` organisation
        train = load_dataset(dataset_id, language, split="train", use_auth_token=True)
        val = load_dataset(dataset_id, language, split="val", use_auth_token=True)
        test = load_dataset(dataset_id, language, split="test", use_auth_token=True)

        # Ensure that the datasets are indeed datasets
        assert isinstance(train, Dataset)
        assert isinstance(val, Dataset)
        assert isinstance(test, Dataset)

        # Merge the splits
        df = concatenate_datasets([train, val, test]).to_pandas()

        # Ensure that `df` is indeed a Pandas DataFrame
        assert isinstance(df, pd.DataFrame)

        # Extract information on which examples contain an answer, as we want to
        # stratify our splits based on this
        has_answer: pd.Series = df.answers.map(lambda dct: dct["text"][0] != "")

        # Split the dataframe into the samples having answers and the samples not
        # having answers
        df_with_answer: pd.DataFrame = df.loc[has_answer]
        df_without_answer: pd.DataFrame = df.loc[~has_answer]

        # Create validation split
        val_size = 256
        val_df_with_answer = df_with_answer.sample(n=val_size // 2, random_state=4242)
        val_df_without_answer = df_without_answer.sample(
            n=val_size // 2, random_state=4242
        )
        val_df = pd.concat([val_df_with_answer, val_df_without_answer])
        val_df = val_df.reset_index(drop=True)

        # Create test split
        test_size = 2048
        df_with_answer_filtered: pd.DataFrame = df_with_answer.loc[
            ~df_with_answer.index.isin(val_df.index)
        ]
        df_without_answer_filtered: pd.DataFrame = df_without_answer.loc[
            ~df_without_answer.index.isin(val_df.index)
        ]
        test_df_with_answer = df_with_answer_filtered.sample(
            n=test_size // 2, random_state=4242
        )
        test_df_without_answer = df_without_answer_filtered.sample(
            n=test_size // 2, random_state=4242
        )
        test_df = pd.concat([test_df_with_answer, test_df_without_answer])
        test_df = test_df.reset_index(drop=True)

        # Create train split
        train_size = 1024
        full_train_df_with_answer = df_with_answer_filtered.loc[
            ~df_with_answer_filtered.index.isin(test_df_with_answer.index)
        ]
        full_train_df_without_answer = df_without_answer_filtered.loc[
            ~df_without_answer_filtered.index.isin(test_df_without_answer.index)
        ]
        train_df_with_answer = full_train_df_with_answer.sample(
            n=train_size // 2, random_state=4242
        )
        train_df_without_answer = full_train_df_without_answer.sample(
            n=train_size // 2, random_state=4242
        )
        train_df = pd.concat([train_df_with_answer, train_df_without_answer])
        train_df = train_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        mini_dataset_id = f"ScandEval/scandiqa-{language}-mini"

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
