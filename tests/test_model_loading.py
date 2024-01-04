"""Unit tests for the `model_loading` module."""

import pytest
from scandeval.dataset_configs import MMLU_DA_CONFIG, SPEED_CONFIG
from scandeval.exceptions import InvalidBenchmark
from scandeval.model_config import get_model_config
from scandeval.model_loading import load_model
from scandeval.utils import GENERATIVE_DATASET_SUPERTASKS, GENERATIVE_DATASET_TASKS


@pytest.mark.parametrize(
    argnames="dataset_config",
    argvalues=[SPEED_CONFIG, MMLU_DA_CONFIG],
    ids=["non-generative", "generative"],
)
def test_load_model(model_id, dataset_config, benchmark_config):
    model_config = get_model_config(
        model_id=model_id, benchmark_config=benchmark_config
    )
    if not (
        dataset_config.task.supertask in GENERATIVE_DATASET_SUPERTASKS
        or dataset_config.task.name in GENERATIVE_DATASET_TASKS
    ):
        tokenizer, model = load_model(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        assert tokenizer is not None
        assert model is not None
    else:
        with pytest.raises(InvalidBenchmark):
            load_model(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )


def test_load_model_with_invalid_model_id(benchmark_config):
    model_config = get_model_config(
        model_id="invalid_model_id", benchmark_config=benchmark_config
    )
    with pytest.raises(InvalidBenchmark):
        load_model(
            model_config=model_config,
            dataset_config=SPEED_CONFIG,
            benchmark_config=benchmark_config,
        )
