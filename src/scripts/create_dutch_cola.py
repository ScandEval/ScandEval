"""Create the Dutch CoLA dataset and upload it to the HF Hub."""

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Dutch CoLA dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "GroNLP/dutch-cola"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = (
        dataset.select_columns(["Sentence", "Acceptability"])
        .rename_columns({"Sentence": "text", "Acceptability": "label"})
        .shuffle(4242)
    )

    label_mapping = {0: "incorrect", 1: "correct"}
    dataset = dataset.map(lambda sample: {"label": label_mapping[sample["label"]]})
    dataset["val"] = dataset.pop("validation")

    # Create dataset ID
    dataset_id = "ScandEval/dutch-cola"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
