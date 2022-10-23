"""Fills in missing model metadata in the stored score JSONL file."""

import json
from pathlib import Path
from typing import Dict, List, Union

from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.model_loading import load_model
from scandeval.types import SCORE_DICT


def main() -> None:
    """Fills in missing model metadata in the stored score JSONL file."""

    # Define list of desired metadata
    metadata = [
        "num_model_parameters",
        "vocab_size",
        "max_sequence_length",
    ]

    # Get mapping from dataset name to dataset configuration
    dataset_configs = get_all_dataset_configs()

    # Create empty list which will contain the modified records
    new_records: List[Dict[str, Union[str, int, List[str], SCORE_DICT]]] = list()

    # Iterate over the records and build new list of records with all metadata
    with Path("scandeval_benchmark_results.jsonl").open() as f:
        for line in f:

            # Load the record
            record = json.loads(line)

            # Get the associated dataset configuration
            dataset_config = dataset_configs[record["dataset"]]

            # Check if the record has missing metadata
            if any([key not in record for key in metadata]):

                # Load the tokenizer and model
                tokenizer, model = load_model(
                    model_id=record["model_id"],
                    revision="main",
                    supertask=dataset_config.task.supertask,
                    num_labels=dataset_config.num_labels,
                    id2label=dataset_config.id2label,
                    label2id=dataset_config.label2id,
                    from_flax=False,
                    use_auth_token=True,
                    cache_dir=".scandeval_cache",
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

            # Store the record
            new_records.append(record)

    # Replace the old file with the new one
    with Path("scandeval_benchmark_results.jsonl").open("w") as f:
        json_str = "\n".join([json.dumps(record) for record in new_records])
        f.write(json_str)


if __name__ == "__main__":
    main()
