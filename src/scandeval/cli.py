"""Command-line interface for benchmarking."""

from typing import List, Tuple

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
            specified then all models will be benchmarked, filtered by `model_language`
            and `model_task`. The specific model version to use can be added after the
            suffix "@": "<model_id>@v1.0.0". It can be a branch name, a tag name, or a
            commit id (currently only supported for Hugging Face models, and it defaults
            to "main" for latest).""",
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
            `model-id` and `dataset` have not both been specified.""",
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
            specified. If "all" then models will not be filtered according to their
            language. If not specified then this will use the `language` value.""",
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
            specified. If "all" then datasets will not be filtered according to their
            language. If not specified then this will use the `language` value.""",
)
@click.option(
    "--model-task",
    "-mt",
    default=None,
    show_default=True,
    multiple=True,
    type=str,
    help="""The model tasks to consider. If not specified then models will not be
            filtered according to the task they were trained on.""",
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
    model_task: Tuple[str],
    dataset_task: Tuple[str],
    evaluate_train: bool,
    no_progress_bar: bool,
    no_save_results: bool,
    cache_dir: str,
    use_auth_token: bool,
    verbose: bool = False,
):
    """Benchmark pretrained language models on Scandinavian language tasks."""

    # Set up language variables
    model_ids = None if len(model_id) == 0 else list(model_id)
    datasets = None if len(dataset) == 0 else list(dataset)
    languages: List[str] = list(language)
    model_languages = None if len(model_language) == 0 else list(model_language)
    dataset_languages = None if len(dataset_language) == 0 else list(dataset_language)
    model_tasks = None if len(model_task) == 0 else list(model_task)
    dataset_tasks = None if len(dataset_task) == 0 else list(dataset_task)

    # Initialise the benchmarker class
    benchmarker = Benchmarker(
        language=languages,
        model_language=model_languages,
        dataset_language=dataset_languages,
        model_task=model_tasks,
        dataset_task=dataset_tasks,
        progress_bar=(not no_progress_bar),
        save_results=(not no_save_results),
        evaluate_train=evaluate_train,
        raise_error_on_invalid_model=raise_error_on_invalid_model,
        verbose=verbose,
        use_auth_token=use_auth_token,
        cache_dir=cache_dir,
    )

    # Perform the benchmark evaluation
    benchmarker(model_id=model_ids, dataset=datasets)
