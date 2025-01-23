"""Create mideind/icelandic_qa_scandeval as a knowledge task dataset and upload it to the HF Hub."""

import json
import os
import random
import re

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from openai import OpenAI
from requests import HTTPError

LABELS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the icelandic knowledge dataset."""
    # Define the base download URL
    repo_id = "mideind/icelandic_qa_scandeval"

    # Download the dataset
    dataset = load_dataset(path=repo_id, split="train")

    dataset = drop_duplicate_questions(dataset)
    assert isinstance(dataset, Dataset)

    # Build the knowledge dataset using a language model
    df = build_dataset_with_llm(dataset=dataset)

    # Create splits
    val_size = 128
    test_size = 1024

    val_df = df.sample(val_size, random_state=42)
    df = df.drop(val_df.index)

    test_df = df.sample(test_size, random_state=42)
    df = df.drop(test_df.index)

    train_df = df
    assert len(train_df) > 800, "The training set should have at least 800 samples."

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
    dataset_id = "ScandEval/icelandic-knowledge"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def drop_duplicate_questions(dataset: Dataset) -> Dataset:
    """Drop duplicate questions from the dataset.

    Args:
        dataset:
            The dataset to drop duplicates from.

    Returns:
        The dataset without duplicates.
    """
    df = dataset.to_pandas()

    # Strip all leading and trailing whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove trailing periods
    df["answer"] = df["answer"].str.rstrip(".")

    # Drop duplicates
    df = df.drop_duplicates(subset="question")

    return Dataset.from_pandas(df)


def build_dataset_with_llm(dataset: Dataset) -> pd.DataFrame:
    """Build the knowledge dataset using a language model.

    Args:
        dataset:
            The dataset to build the knowledge dataset from.

    Returns:
        The knowledge dataset.
    """
    df = dataset.to_pandas()

    assert isinstance(df, pd.DataFrame)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    cache_file = "icelandic_qa_scandeval_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    texts: list[str] = []
    correct_labels: list[str] = []
    for i, row in df.iterrows():
        id_ = str(i)
        if id_ not in cache:
            messages = [
                {
                    "role": "user",
                    "content": f"For the question: {row.question} where the correct answer is: {row.answer}, please provide 3 plausible alternatives in Icelandic. Format your response using XML tags: <option1>first</option1> <option2>second</option2> <option3>third</option3>",
                }
            ]

            response = client.chat.completions.create(model="gpt-4", messages=messages)

            # Store response
            cache[id_] = response.choices[0].message.content
            with open(cache_file, "w") as f:
                json.dump(cache, f)

        # Extract options
        content = cache[id_]

        option1 = _extract_option(option="option1", content=content)
        option2 = _extract_option(option="option2", content=content)
        option3 = _extract_option(option="option3", content=content)

        random.shuffle(LABELS)
        options = {
            LABELS[0]: option1,
            LABELS[1]: option2,
            LABELS[2]: option3,
            LABELS[3]: row.answer,
        }
        assert (
            len(set(options.values())) == 4
        ), f"Expected 4 unique options, but got {options}"
        correct_label = [k for k, v in options.items() if v == row.answer][0]

        text = f"{row.question}\nSvarmöguleikar:\na. {options['a']}\nb. {options['b']}\nc. {options['c']}\nd. {options['d']}"

        texts.append(text)
        correct_labels.append(correct_label)

    df_llm = pd.DataFrame({"text": texts, "label": correct_labels})
    return df_llm


def _extract_option(option: str, content: str) -> str:
    """Extract an option from the LLM generated content.

    Args:
        option:
            The option to extract.
        content:
            The content to extract the option from.

    Returns:
        The extracted option.
    """
    if match := re.search(rf"<{option}>(.*?)</{option}>", content):
        return match.group(1)
    raise ValueError(f"Option {option} not found in response: {content}")


if __name__ == "__main__":
    main()
