"""Create the ARC-mini datasets and upload them to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the ARC-mini datasets and upload them to the HF Hub."""

    # Define the base download URL
    repo_id = "alexandrainst/m_arc"

    # Create a mapping with the word "Choices" in different languages
    choices_mapping = dict(
        da="Svarmuligheder",
        sv="Svarsalternativ",
        de="Antwortmöglichkeiten",
        nl="Antwoordopties",
    )

    for language in ["da", "sv", "de", "nl"]:
        # Download the dataset
        dataset = load_dataset(path=repo_id, name=language, token=True)
        assert isinstance(dataset, DatasetDict)

        # Convert the dataset to a dataframe
        train_df = dataset["train"].to_pandas()
        val_df = dataset["val"].to_pandas()
        test_df = dataset["test"].to_pandas()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        # Rename the columns
        train_df.rename(columns=dict(answer="label"), inplace=True)
        val_df.rename(columns=dict(answer="label"), inplace=True)
        test_df.rename(columns=dict(answer="label"), inplace=True)

        # Remove all samples with a non-null value of `option_e`
        train_df = train_df[train_df["option_e"].isnull()]
        val_df = val_df[val_df["option_e"].isnull()]
        test_df = test_df[test_df["option_e"].isnull()]

        # Remove all samples with a null value of `option_a`, `option_b`, `option_c` or
        # `option_d`
        train_df = train_df[
            train_df["option_a"].notnull()
            & train_df["option_b"].notnull()
            & train_df["option_c"].notnull()
            & train_df["option_d"].notnull()
        ]
        val_df = val_df[
            val_df["option_a"].notnull()
            & val_df["option_b"].notnull()
            & val_df["option_c"].notnull()
            & val_df["option_d"].notnull()
        ]
        test_df = test_df[
            test_df["option_a"].notnull()
            & test_df["option_b"].notnull()
            & test_df["option_c"].notnull()
            & test_df["option_d"].notnull()
        ]

        # Make a `text` column with all the options in it
        train_df["text"] = [
            row.instruction.replace("\n", " ").strip() + "\n"
            f"{choices_mapping[language]}:\n"
            f"a. " + row.option_a.replace("\n", " ").strip() + "\n"
            "b. " + row.option_b.replace("\n", " ").strip() + "\n"
            "c. " + row.option_c.replace("\n", " ").strip() + "\n"
            "d. " + row.option_d.replace("\n", " ").strip()
            for _, row in train_df.iterrows()
        ]
        val_df["text"] = [
            row.instruction.replace("\n", " ").strip() + "\n"
            f"{choices_mapping[language]}:\n"
            f"a. " + row.option_a.replace("\n", " ").strip() + "\n"
            "b. " + row.option_b.replace("\n", " ").strip() + "\n"
            "c. " + row.option_c.replace("\n", " ").strip() + "\n"
            "d. " + row.option_d.replace("\n", " ").strip()
            for _, row in val_df.iterrows()
        ]
        test_df["text"] = [
            row.instruction.replace("\n", " ").strip() + "\n"
            f"{choices_mapping[language]}:\n"
            f"a. " + row.option_a.replace("\n", " ").strip() + "\n"
            "b. " + row.option_b.replace("\n", " ").strip() + "\n"
            "c. " + row.option_c.replace("\n", " ").strip() + "\n"
            "d. " + row.option_d.replace("\n", " ").strip()
            for _, row in test_df.iterrows()
        ]

        # Only keep the `text` and `label` columns
        train_df = train_df[["text", "label"]]
        val_df = val_df[["text", "label"]]
        test_df = test_df[["text", "label"]]

        # Remove duplicates
        train_df.drop_duplicates(inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        val_df.drop_duplicates(inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.drop_duplicates(inplace=True)
        test_df.reset_index(drop=True, inplace=True)

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
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        dataset_id = f"ScandEval/arc-{language}-mini"

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
