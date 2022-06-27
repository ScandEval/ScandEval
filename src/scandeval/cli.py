"""Command-line interface for benchmarking."""

from typing import List, Optional, Tuple

import click

from .benchmarker import Benchmarker
from .datasets import get_all_dataset_configs


@click.command()
@click.option(
    "--model-id",
    "-m",
    default=None,
    show_default=True,
    multiple=True,
    help="""The HuggingFace model ID of the model(s) to be benchmarked. If not
            specified then all models will be benchmarked, filtered by `language` and
            `task`. The specific model version to use can be added after the suffix
            "@": "<model_id>@v1.0.0". It can be a branch name, a tag name, or a commit
            id (currently only supported for HuggingFace models, and it defaults to
            "main" for latest).""",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    show_default=True,
    multiple=True,
    type=click.Choice([config.name for config in get_all_dataset_configs()]),
    help="""The name of the benchmark dataset. If not specified then all datasets will
            be benchmarked.""",
)
@click.option(
    "--language",
    "-l",
    default=["da", "sv", "no"],
    show_default=True,
    multiple=True,
    type=click.Choice(["da", "sv", "no", "nb", "nn", "is", "fo"]),
    help="""The languages to benchmark, both for models and datasets. Only relevant if
            `model-id` and `dataset` have not both been specified.""",
)
@click.option(
    "--model-language",
    "-ml",
    default=None,
    show_default=True,
    multiple=True,
    type=click.Choice(["da", "sv", "no", "nb", "nn", "is", "fo"]),
    help="""The model languages to benchmark. Only relevant if `model-id` has not been
            specified. If specified then this will override `language` for models.""",
)
@click.option(
    "--dataset-language",
    "-dl",
    default=None,
    show_default=True,
    multiple=True,
    type=click.Choice(["da", "sv", "no", "nb", "nn", "is", "fo"]),
    help="""The dataset languages to benchmark. Only relevant if `dataset` has not been
            specified. If specified then this will override `language` for datasets.""",
)
@click.option(
    "--task",
    "-t",
    default=["all"],
    show_default=True,
    multiple=True,
    type=click.Choice(
        ["all", "fill-mask", "token-classification", "text-classification"]
    ),
    help="The tasks to benchmark. Only relevant if `model-id` " "is not specified.",
)
@click.option(
    "--evaluate-train",
    "-e",
    is_flag=True,
    show_default=True,
    help="Whether the training set should be evaluated.",
)
@click.option(
    "--no-progress-bar",
    "-np",
    is_flag=True,
    show_default=True,
    help="Whether progress bars should be shown.",
)
@click.option(
    "--raise-error-on-invalid-model",
    "-r",
    is_flag=True,
    show_default=True,
    help="Whether to raise an error if a model is invalid.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    show_default=True,
    help="Whether extra input should be outputted during benchmarking",
)
@click.option(
    "--no-save-results",
    "-ns",
    is_flag=True,
    show_default=True,
    help="Whether results should not be stored to disk.",
)
@click.option(
    "--cache-dir",
    "-c",
    default=".scandeval_cache",
    show_default=True,
    help="The directory where models are datasets are cached.",
)
@click.option(
    "--use-auth-token",
    is_flag=True,
    show_default=True,
    help="""Whether an authentication token should be used, enabling evaluation of
            private models. Requires that you are logged in via the
            `huggingface-cli login` command.""",
)
def benchmark(
    model_id: Tuple[str],
    dataset: Tuple[str],
    language: Tuple[str],
    model_language: Tuple[str],
    dataset_language: Tuple[str],
    raise_error_on_invalid_model: bool,
    task: Tuple[str],
    evaluate_train: bool,
    no_progress_bar: bool,
    no_save_results: bool,
    cache_dir: str,
    use_auth_token: bool,
    verbose: bool = False,
):
    """Benchmark language models on Scandinavian language tasks."""

    # Set up variables
    languages: List[str] = list(language)
    model_languages: Optional[List[str]]
    dataset_languages: Optional[List[str]]
    if len(model_language) > 0:
        model_languages = list(model_language)
    else:
        model_languages = None
    if len(dataset_language) > 0:
        dataset_languages = list(dataset_language)
    else:
        dataset_languages = None
    tasks = "all" if "all" in task else list(task)

    # Initialise the benchmarker class
    benchmarker = Benchmarker(
        language=languages,
        model_language=model_languages,
        dataset_language=dataset_languages,
        task=tasks,
        progress_bar=(not no_progress_bar),
        save_results=(not no_save_results),
        evaluate_train=evaluate_train,
        raise_error_on_invalid_model=raise_error_on_invalid_model,
        verbose=verbose,
        use_auth_token=use_auth_token,
        cache_dir=cache_dir,
    )

    # Perform the benchmark evaluation
    model_id_list = None if len(model_id) == 0 else list(model_id)
    dataset_list = None if len(dataset) == 0 else list(dataset)
    benchmarker(model_id=model_id_list, dataset=dataset_list)
