"""Create the SQuAD-mini datasets and upload them to the HF Hub."""

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the SQuAD-mini datasets and upload them to the HF Hub."""

    dataset_id = "squad_v2"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    valtest_df = dataset["validation"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(valtest_df, pd.DataFrame)

    # Extract information on which examples contain an answer
    def has_answer(example: dict) -> bool:
        return len(example["text"]) > 0 and example["text"][0] != ""

    train_has_answer: pd.Series = train_df.answers.map(has_answer)
    valtest_has_answer: pd.Series = valtest_df.answers.map(has_answer)

    # Only work with the questions having answers in the context
    train_df_with_answer: pd.DataFrame = train_df.loc[train_has_answer]
    valtest_df_with_answer: pd.DataFrame = valtest_df.loc[valtest_has_answer]

    # Create validation split
    val_size = 256
    val_df = valtest_df_with_answer.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 2048
    valtest_df_with_answer_filtered: pd.DataFrame = valtest_df_with_answer.loc[
        ~valtest_df_with_answer.index.isin(val_df.index)
    ]
    test_df = valtest_df_with_answer_filtered.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df_with_answer.sample(n=train_size, random_state=4242)

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
    mini_dataset_id = "ScandEval/squad-mini"

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