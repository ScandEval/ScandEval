"""All benchmarks tasks used in ScandEval."""

from typing import Dict

from .config import DatasetTask, MetricConfig


def get_all_dataset_tasks() -> Dict[str, DatasetTask]:
    """Get a list of all the dataset tasks.

    Returns:
        dict:
            A mapping between names of dataset tasks and their configurations.
    """
    return {cfg.name: cfg for cfg in globals().values() if isinstance(cfg, DatasetTask)}


LA = DatasetTask(
    name="linguistic-acceptability",
    supertask="sequence-classification",
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="macro_f1",
            pretty_name="Macro-average F1-score",
            huggingface_id="f1",
            results_key="f1",
            compute_kwargs=dict(average="macro"),
        ),
    ],
    labels=["INCORRECT", "CORRECT"],
)


NER = DatasetTask(
    name="named-entity-recognition",
    supertask="token-classification",
    metrics=[
        MetricConfig(
            name="micro_f1",
            pretty_name="Micro-average F1-score",
            huggingface_id="seqeval",
            results_key="overall_f1",
        ),
        MetricConfig(
            name="micro_f1_no_misc",
            pretty_name="Micro-average F1-score without MISC tags",
            huggingface_id="seqeval",
            results_key="overall_f1",
        ),
    ],
    labels=[
        "O",
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-PER",
        "I-PER",
        "B-MISC",
        "I-MISC",
    ],
)


QA = DatasetTask(
    name="question-answering",
    supertask="question-answering",
    metrics=[
        MetricConfig(
            name="em",
            pretty_name="Exact Match",
            huggingface_id="squad_v2",
            results_key="exact",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2f}%"),
        ),
        MetricConfig(
            name="f1",
            pretty_name="F1-score",
            huggingface_id="squad_v2",
            results_key="f1",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2f}%"),
        ),
    ],
    labels=["START_POSITIONS", "END_POSITIONS"],
)


SENT = DatasetTask(
    name="sentiment-classification",
    supertask="sequence-classification",
    metrics=[
        MetricConfig(
            name="mcc",
            pretty_name="Matthew's Correlation Coefficient",
            huggingface_id="matthews_correlation",
            results_key="matthews_correlation",
        ),
        MetricConfig(
            name="macro_f1",
            pretty_name="Macro-average F1-score",
            huggingface_id="f1",
            results_key="f1",
            compute_kwargs=dict(average="macro"),
        ),
    ],
    labels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
)


SPEED = DatasetTask(
    name="speed",
    supertask="sequence-classification",
    metrics=[
        MetricConfig(
            name="speed",
            pretty_name="Inferences per second",
            huggingface_id="",
            results_key="speed",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2f}"),
        ),
    ],
    labels=[],
)
