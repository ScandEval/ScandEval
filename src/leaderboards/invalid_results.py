"""Public list of (model, dataset) results left out of aggregated scores.

Some models have been trained on datasets that EuroEval evaluates on, which
invalidates their score on those datasets. Such results are listed in
`invalid_results.yaml`. A listed result is still shown on the leaderboards, marked as
invalid, but it does not count towards the model's aggregated score or rank.
"""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
from pathlib import Path

from yaml import safe_load

from euroeval.date_utils import normalise_release_date

from .constants import INVALID_RESULTS_CONFIG
from .records import plain_model_id
from .task_metadata import all_dataset_names


@dataclasses.dataclass(frozen=True)
class InvalidResult:
    """A (model, dataset) result left out of the model's aggregated score and rank.

    Attributes:
        model:
            The model ID, without revision or parameter suffixes. The entry covers
            every variant of the model.
        dataset:
            The dataset name.
        reason:
            Why the result is invalid.
        source:
            Where the evidence for the reason can be found.
        date:
            When the result was marked as invalid.
    """

    model: str
    dataset: str
    reason: str
    source: str
    date: dt.date


def invalid_datasets_by_model(
    invalid_results: c.Sequence[InvalidResult], model_ids: c.Iterable[str]
) -> dict[str, set[str]]:
    """Map each result model ID to the datasets on which its result is invalid.

    An entry matches a model ID when they are equal after removing the
    `(zero-shot)`/`(val)` variant suffix, or when the model ID continues with an
    `@revision` or `#parameter` suffix.

    Args:
        invalid_results:
            The invalid results, as returned by `load_invalid_results`.
        model_ids:
            The model IDs of all loaded results.

    Returns:
        A mapping from model ID to its invalid datasets. Model IDs without invalid
        results are left out.

    Raises:
        ValueError:
            If an entry matches none of the model IDs.
    """
    invalid_datasets: dict[str, set[str]] = {}
    unmatched_models = {result.model for result in invalid_results}
    for model_id in model_ids:
        plain_id = plain_model_id(model_id=model_id)
        for result in invalid_results:
            if plain_id == result.model or plain_id.startswith(
                (f"{result.model}@", f"{result.model}#")
            ):
                invalid_datasets.setdefault(model_id, set()).add(result.dataset)
                unmatched_models.discard(result.model)
    if unmatched_models:
        raise ValueError(
            "The invalid results refer to models without any loaded results: "
            f"{', '.join(sorted(unmatched_models))}."
        )
    return invalid_datasets


def load_invalid_results(path: Path = INVALID_RESULTS_CONFIG) -> list[InvalidResult]:
    """Load and validate the invalid (model, dataset) results.

    Args:
        path (optional):
            The YAML file to load. Defaults to the bundled `invalid_results.yaml`.

    Returns:
        The invalid results. Empty if the file contains no entries.

    Raises:
        ValueError:
            If the file is not a list of entries, an entry has missing or unknown
            fields, or an entry refers to a dataset that EuroEval does not define.
    """
    with path.open(encoding="utf-8") as f:
        data = safe_load(f)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a list of entries, but got {type(data).__name__}."
        )
    invalid_results = [
        _parse_entry(entry=entry, index=index, path=path)
        for index, entry in enumerate(data, start=1)
    ]
    known_datasets = all_dataset_names()
    unknown_datasets = sorted(
        {result.dataset for result in invalid_results} - known_datasets
    )
    if unknown_datasets:
        raise ValueError(
            f"{path} refers to datasets that EuroEval does not define: "
            f"{', '.join(unknown_datasets)}."
        )
    return invalid_results


def _parse_entry(entry: object, index: int, path: Path) -> InvalidResult:
    """Parse a single entry of the invalid results file.

    Args:
        entry:
            The raw YAML entry.
        index:
            The 1-based position of the entry, used in error messages.
        path:
            The file the entry was loaded from, used in error messages.

    Returns:
        The parsed entry.

    Raises:
        ValueError:
            If the entry is not a mapping, has missing or unknown fields, or has an
            empty or malformed value.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Entry {index} in {path} must be a mapping.")
    fields = [field.name for field in dataclasses.fields(InvalidResult)]
    missing_fields = [field for field in fields if field not in entry]
    unknown_fields = sorted(str(key) for key in entry if key not in fields)
    if missing_fields or unknown_fields:
        raise ValueError(
            f"Entry {index} in {path} has missing fields {missing_fields} and "
            f"unknown fields {unknown_fields}. The fields must be exactly {fields}."
        )
    date = entry["date"]
    if isinstance(date, dt.datetime) or normalise_release_date(value=date) is None:
        raise ValueError(
            f"Entry {index} in {path} has a date that is not YYYY-MM-DD: {date!r}."
        )
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    text_values = {field: entry[field] for field in fields if field != "date"}
    empty_fields = [
        field
        for field, value in text_values.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if empty_fields:
        raise ValueError(
            f"Entry {index} in {path} must have non-empty text in {empty_fields}."
        )
    return InvalidResult(
        model=text_values["model"].strip(),
        dataset=text_values["dataset"].strip(),
        reason=text_values["reason"].strip(),
        source=text_values["source"].strip(),
        date=date,
    )
