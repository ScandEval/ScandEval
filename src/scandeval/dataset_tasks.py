"""All benchmarks tasks used in ScandEval."""

from .config import DatasetTask, MetricConfig


def get_all_dataset_tasks() -> dict[str, DatasetTask]:
    """Get a list of all the dataset tasks.

    Returns:
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
    labels=["incorrect", "correct"],
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
        "o",
        "b-loc",
        "i-loc",
        "b-org",
        "i-org",
        "b-per",
        "i-per",
        "b-misc",
        "i-misc",
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
    labels=["start_positions", "end_positions"],
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
    labels=["negative", "neutral", "positive"],
)


SUMMARIZATION = DatasetTask(
    name="summarization",
    supertask="text-to-text",
    metrics=[
        MetricConfig(
            name="bertscore",
            pretty_name="BERTScore",
            huggingface_id="bertscore",
            results_key="f1",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2%}"),
            compute_kwargs=dict(model_type="microsoft/mdeberta-v3-base", device="cpu"),
        ),
        MetricConfig(
            name="rouge-1",
            pretty_name="ROUGE-1",
            huggingface_id="rouge",
            results_key="rouge1",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2%}"),
        ),
        MetricConfig(
            name="rouge-2",
            pretty_name="ROUGE-2",
            huggingface_id="rouge",
            results_key="rouge2",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2%}"),
        ),
        MetricConfig(
            name="rouge-l",
            pretty_name="ROUGE-L",
            huggingface_id="rouge",
            results_key="rougeL",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2%}"),
        ),
        MetricConfig(
            name="bleu",
            pretty_name="BLEU",
            huggingface_id="bleu",
            results_key="bleu",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2%}"),
        ),
        MetricConfig(
            name="chrf++",
            pretty_name="chrF++",
            huggingface_id="chrf",
            results_key="score",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2f}%"),
            compute_kwargs=dict(word_order=2),
        ),
        MetricConfig(
            name="character",
            pretty_name="CharacTER",
            huggingface_id="character",
            results_key="cer_score",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:.2%}"),
        ),
    ],
    labels=[],
)


MULTIPLE_CHOICE = DatasetTask(
    name="multiple-choice",
    supertask="multiple-choice",
    metrics=[  # Not sure
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
    labels=[],  # Maybe add?
)


TEXT_MODELLING = DatasetTask(
    name="text-modelling",
    supertask="text-modelling",
    metrics=[
        MetricConfig(
            name="perplexity",
            pretty_name="Perplexity",
            huggingface_id="perplexity",
            results_key="mean_perplexity",
        ),
    ],
    labels=[],
)


SPEED = DatasetTask(
    name="speed",
    supertask="sequence-classification",
    metrics=[
        MetricConfig(
            name="speed",
            pretty_name="Tokens per second",
            huggingface_id="",
            results_key="speed",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        ),
        MetricConfig(
            name="speed_short",
            pretty_name="Tokens per second on short documents",
            huggingface_id="",
            results_key="speed",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        ),
    ],
    labels=[],
)
