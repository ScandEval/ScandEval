"""Create the Dutch CoLA test set and upload it to the HF Hub."""

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the DutchSocial-mini sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "GroNLP/dutch-cola"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    del dataset["train"]
    assert isinstance(dataset, DatasetDict)

    dataset = dataset.select_columns(["Sentence", "Acceptability"]).rename_columns({"Sentence": "text", "Acceptability": "label"})

    label_mapping = {0: "incorrect", 1: "correct"}
    dataset = dataset.map(lambda sample: {"label": label_mapping[sample["label"]]})

    # Create dataset ID
    dataset_id = "BramVanroy/dutch-cola"

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
