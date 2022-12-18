"""Command-line interface for benchmarking."""

from typing import List, Tuple, Union

import click

from .benchmarker import Benchmarker
from .dataset_configs import get_all_dataset_configs
from .dataset_tasks import get_all_dataset_tasks
from .languages import get_all_languages


@click.command()
@click.option(
    "--model-id",
    "-m",
    default=None,
    show_default=True,
    multiple=True,
    help="""The Hugging Face model ID of the model(s) to be benchmarked. If not
    specified then all models will be benchmarked, filtered by `model_language` and
    `model_task`. The specific model version to use can be added after the suffix "@":
    "<model_id>@v1.0.0". It can be a branch name, a tag name, or a commit id, and
    defaults to the latest version if not specified.""",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    show_default=True,
    multiple=True,
    type=click.Choice(list(get_all_dataset_configs().keys())),
    help="""The name of the benchmark dataset. If not specified then all datasets will
    be benchmarked, filtered by `dataset_language` and `dataset_task`.""",
)
@click.option(
    "--language",
    "-l",
    default=["da", "sv", "no"],
    show_default=True,
    multiple=True,
    metavar="ISO 639-1 LANGUAGE CODE",
    type=click.Choice(["all"] + list(get_all_languages().keys())),
    help="""The languages to benchmark, both for models and datasets. Only relevant if
    `model-id` and `dataset` have not both been specified. If "all" then all models
    will be benchmarked on all datasets.""",
)
@click.option(
    "--model-language",
    "-ml",
    default=None,
    show_default=True,
    multiple=True,
    metavar="ISO 639-1 LANGUAGE CODE",
    type=click.Choice(["all"] + list(get_all_languages().keys())),
    help="""The model languages to benchmark. Only relevant if `model-id` has not been
    specified. If "all" then all models will be benchmarked. If not specified then this
    will use the `language` value.""",
)
@click.option(
    "--dataset-language",
    "-dl",
    default=None,
    show_default=True,
    multiple=True,
    metavar="ISO 639-1 LANGUAGE CODE",
    type=click.Choice(["all"] + list(get_all_languages().keys())),
    help="""The dataset languages to benchmark. Only relevant if `dataset` has not been
    specified. If "all" then the models will be benchmarked on all datasets. If not
    specified then this will use the `language` value.""",
)
@click.option(
    "--dataset-task",
    "-dt",
    default=None,
    show_default=True,
    multiple=True,
    type=click.Choice(list(get_all_dataset_tasks().keys())),
    help="The dataset tasks to consider.",
)
@click.option(
    "--batch-size",
    default="32",
    type=click.Choice(["1", "2", "4", "8", "16", "32"]),
    help="The batch size to use.",
)
@click.option(
    "--evaluate-train/--no-evaluate-train",
    default=False,
    show_default=True,
    help="Whether to evaluate the model on the training set.",
)
@click.option(
    "--progress-bar/--no-progress-bar",
    "-p/-np",
    default=True,
    show_default=True,
    help="Whether to show a progress bar.",
)
@click.option(
    "--raise-error-on-invalid-model/--no-raise-error-on-invalid-model",
    default=False,
    show_default=True,
    help="Whether to raise an error if a model is invalid.",
)
@click.option(
    "--verbose/--no-verbose",
    "-v/-nv",
    default=False,
    show_default=True,
    help="Whether extra input should be outputted during benchmarking",
)
@click.option(
    "--save-results/--no-save-results",
    "-s/-ns",
    default=True,
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
    "--auth-token",
    type=str,
    default="",
    show_default=True,
    help="""The authentication token for the Hugging Face Hub. If specified then the
    `--use-auth-token` flag will be set to True.""",
)
@click.option(
    "--use-auth-token/--no-use-auth-token",
    default=False,
    show_default=True,
    help="""Whether an authentication token should be used, enabling evaluation of
    private models. Requires that you are logged in via the `huggingface-cli login`
    command.""",
)
@click.option(
    "--ignore-duplicates/--no-ignore-duplicates",
    default=True,
    show_default=True,
    help="""Whether to skip evaluation of models which have already been evaluated,
    with scores lying in the 'scandeval_benchmark_results.jsonl' file.""",
)
def benchmark(
    model_id: Tuple[str],
    dataset: Tuple[str],
    language: Tuple[str],
    model_language: Tuple[str],
    dataset_language: Tuple[str],
    raise_error_on_invalid_model: bool,
    dataset_task: Tuple[str],
    batch_size: str,
    evaluate_train: bool,
    progress_bar: bool,
    save_results: bool,
    cache_dir: str,
    use_auth_token: bool,
    auth_token: str,
    ignore_duplicates: bool,
    verbose: bool = False,
) -> None:
    """Benchmark pretrained language models on Scandinavian language tasks."""

    # Set up language variables
    model_ids = None if len(model_id) == 0 else list(model_id)
    datasets = None if len(dataset) == 0 else list(dataset)
    languages: List[str] = list(language)
    model_languages = None if len(model_language) == 0 else list(model_language)
    dataset_languages = None if len(dataset_language) == 0 else list(dataset_language)
    dataset_tasks = None if len(dataset_task) == 0 else list(dataset_task)
    batch_size_int = int(batch_size)
    auth: Union[str, bool] = auth_token if auth_token != "" else use_auth_token

    # Initialise the benchmarker class
    benchmarker = Benchmarker(
        language=languages,
        model_language=model_languages,
        dataset_language=dataset_languages,
        dataset_task=dataset_tasks,
        batch_size=batch_size_int,
        progress_bar=progress_bar,
        save_results=save_results,
        evaluate_train=evaluate_train,
        raise_error_on_invalid_model=raise_error_on_invalid_model,
        verbose=verbose,
        use_auth_token=auth,
        ignore_duplicates=ignore_duplicates,
        cache_dir=cache_dir,
    )

    # Perform the benchmark evaluation
    benchmarker(model_id=model_ids, dataset=datasets)
