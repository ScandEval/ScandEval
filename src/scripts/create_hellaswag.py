"""Create the HellaSwag-mini datasets and upload them to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the HellaSwag-mini datasets and upload them to the HF Hub."""

    # Define the base download URL
    repo_id = "alexandrainst/m_hellaswag"

    # Create a mapping with the word "Choices" in different languages
    choices_mapping = dict(
        da="Svarmuligheder",
        sv="Svarsalternativ",
        de="Antwortmöglichkeiten",
        nl="Antwoordopties",
    )

    for language in ["da", "sv", "de", "nl"]:
        # Download the dataset
        dataset = load_dataset(path=repo_id, name=language, token=True, split="val")
        assert isinstance(dataset, Dataset)

        # Convert the dataset to a dataframe
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Make a `text` column with all the options in it
        df["text"] = [
            f"{row['ctx']}\n"
            f"{choices_mapping[language]}:\n"
            f"A: {row['endings'][0]}\n"
            f"B: {row['endings'][1]}\n"
            f"C: {row['endings'][2]}\n"
            f"D: {row['endings'][3]}"
            for _, row in df.iterrows()
        ]

        # Fix the label column
        label_mapping = {"0": "A", "1": "B", "2": "C", "3": "D"}
        df.label = df.label.map(label_mapping)

        # Only keep the samples whose `activity_label` has at least 3 samples
        acceptable_activity_labels = [
            activity_label
            for activity_label, count in df["activity_label"].value_counts().items()
            if count >= 3
        ]
        df = df[df["activity_label"].isin(acceptable_activity_labels)]

        # Remove duplicates
        df.drop_duplicates(subset="ctx", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Only keep the columns `text`, `label` and `activity_label`
        df = df[["text", "label", "activity_label"]]

        # Create validation split
        val_size = 256
        traintest_arr, val_arr = train_test_split(
            df, test_size=val_size, random_state=4242, stratify=df.activity_label
        )
        traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
        val_df = pd.DataFrame(val_arr, columns=df.columns)

        # Create test split
        test_size = 2048
        train_arr, test_arr = train_test_split(
            traintest_df,
            test_size=test_size,
            random_state=4242,
            stratify=traintest_df.activity_label,
        )
        train_df = pd.DataFrame(train_arr, columns=df.columns)
        test_df = pd.DataFrame(test_arr, columns=df.columns)

        # Create train split
        train_size = 1024
        train_df = train_df.sample(train_size, random_state=4242)

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
        dataset_id = f"ScandEval/hellaswag-{language}-mini"

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
