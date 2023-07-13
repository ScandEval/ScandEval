"""Fills in missing model metadata in the stored score JSONL file."""

import json
from pathlib import Path
from typing import Dict, List, Union

from tqdm.auto import tqdm

from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.model_loading import load_model
from scandeval.types import SCORE_DICT
from scandeval.utils import clear_memory


def main() -> None:
    """Fills in missing model metadata in the stored score JSONL file."""

    # Define list of desired metadata
    metadata = [
        "num_model_parameters",
        "vocabulary_size",
        "max_sequence_length",
    ]

    # Get mapping from dataset name to dataset configuration
    dataset_configs = get_all_dataset_configs()

    # Create empty list which will contain the modified records
    new_records: List[Dict[str, Union[str, int, List[str], SCORE_DICT]]] = list()

    # Create cache for the metadata, to avoid loading in a model too many times
    cache: Dict[str, Dict[str, int]] = dict()

    # Iterate over the records and build new list of records with all metadata
    with Path("scandeval_benchmark_results.jsonl").open() as f:
        lines = f.readlines()

    with tqdm(lines, desc="Adding metadata to records") as pbar:
        for line in pbar:
            # Skip line if it is empty
            if line.strip() == "":
                continue

            # Load the record
            record = json.loads(line)

            # Update pbar description
            pbar.set_description(f"Adding metadata to records ({record['model']})")

            # Get the associated dataset configuration
            dataset_config = dataset_configs[record["dataset"]]

            # Check if scores are of the wrong scale
            total_scores: Dict[str, float] = record["results"]["total"]
            if all([score < 1 for score in total_scores.values()]):
                for key, score in total_scores.items():
                    record["results"]["total"][key] = score * 100

            # Check if the record has missing metadata
            if any([key not in record for key in metadata]):
                # Load metadata from cache if possible
                if record["model"] in cache:
                    cached_metadata = cache[record["model"]]
                    for key, val in cached_metadata.items():
                        record[key] = val

                # Otherwise, load the tokenizer and model and extract the metadata
                else:
                    tokenizer, model = load_model(
                        model_id=record["model"],
                        revision="main",
                        supertask=dataset_config.task.supertask,
                        language=dataset_config.languages[0].code,
                        num_labels=dataset_config.num_labels,
                        id2label=dataset_config.id2label,
                        label2id=dataset_config.label2id,
                        from_flax=False,
                        use_auth_token=True,
                        cache_dir=".scandeval_cache",
                        raise_errors=False,
                    )

                    # Add the metadata to the record
                    record["num_model_parameters"] = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    record["max_sequence_length"] = tokenizer.model_max_length
                    if hasattr(model.config, "vocab_size"):
                        record["vocab_size"] = model.config.vocab_size
                    elif hasattr(tokenizer, "vocab_size"):
                        record["vocab_size"] = tokenizer.vocab_size
                    else:
                        record["vocab_size"] = -1

                    # Add the metadata to the cache
                    cache[record["model"]] = dict(
                        num_model_parameters=record["num_model_parameters"],
                        max_sequence_length=record["max_sequence_length"],
                        vocab_size=record["vocab_size"],
                    )

                    # Clear the tokenizer and model from memory
                    del tokenizer
                    del model
                    clear_memory()

            # Store the record
            new_records.append(record)

    # Replace the old file with the new one
    with Path("scandeval_benchmark_results.jsonl").open("w") as f:
        json_str = "\n".join([json.dumps(record) for record in new_records])
        f.write(json_str)


if __name__ == "__main__":
    main()
