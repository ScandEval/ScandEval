"""Command-line interface for benchmarking."""

import click

from .benchmarker import Benchmarker
from .dataset_configs import get_all_dataset_configs
from .dataset_tasks import get_all_dataset_tasks
from .enums import Device, Framework
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
    "--raise-errors/--no-raise-errors",
    default=False,
    show_default=True,
    help="Whether to raise errors instead of skipping the evaluation.",
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
    "--token",
    type=str,
    default="",
    show_default=True,
    help="""The authentication token for the Hugging Face Hub. If specified then the
    `--token` flag will be set to True.""",
)
@click.option(
    "--use-token/--no-use-token",
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
@click.option(
    "--framework",
    "-f",
    default=None,
    show_default=True,
    type=click.Choice([framework.lower() for framework in Framework.__members__]),
    help="""The model framework to use. Only relevant if `model-id` refers to a local
    path. Otherwise, the framework will be set automatically.""",
)
@click.option(
    "--device",
    default=None,
    show_default=True,
    type=click.Choice([device.lower() for device in Device.__members__]),
    help="""The device to use for evaluation. If not specified then the device will be
    set automatically.""",
)
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=False,
    show_default=True,
    help="""Whether to trust remote code. Only set this flag if you trust the supplier
    of the model.""",
)
@click.option(
    "--load-in-4bit/--no-load-in-4bit",
    default=None,
    show_default=True,
    help="Whether to load the model in 4-bit precision.",
)
@click.option(
    "--use-flash-attention/--no-use-flash-attention",
    default=False,
    show_default=True,
    help="Whether to use Flash Attention.",
)
@click.option(
    "--clear-model-cache/--no-clear-model-cache",
    default=False,
    show_default=True,
    help="Whether to clear the model cache after benchmarking each model.",
)
@click.option(
    "--only-validation-split/--no-only-validation-split",
    default=False,
    show_default=True,
    help="Whether to only evaluate on the validation split.",
)
@click.option(
    "--few-shot/--no-few-shot",
    default=True,
    show_default=True,
    help="Whether to only evaluate the model using few-shot evaluation. Only relevant "
    "if the model is generative.",
)
def benchmark(
    model_id: tuple[str],
    dataset: tuple[str],
    language: tuple[str],
    model_language: tuple[str],
    dataset_language: tuple[str],
    raise_errors: bool,
    dataset_task: tuple[str],
    batch_size: str,
    evaluate_train: bool,
    progress_bar: bool,
    save_results: bool,
    cache_dir: str,
    use_token: bool,
    token: str,
    ignore_duplicates: bool,
    verbose: bool,
    framework: str | None,
    device: str | None,
    trust_remote_code: bool,
    load_in_4bit: bool | None,
    use_flash_attention: bool,
    clear_model_cache: bool,
    only_validation_split: bool,
    few_shot: bool,
) -> None:
    """Benchmark pretrained language models on language tasks."""
    # Set up language variables
    model_ids = None if len(model_id) == 0 else list(model_id)
    datasets = None if len(dataset) == 0 else list(dataset)
    languages: list[str] = list(language)
    model_languages = None if len(model_language) == 0 else list(model_language)
    dataset_languages = None if len(dataset_language) == 0 else list(dataset_language)
    dataset_tasks = None if len(dataset_task) == 0 else list(dataset_task)
    batch_size_int = int(batch_size)
    device = Device[device.upper()] if device is not None else None
    token_str_bool: str | bool = token if token != "" else use_token

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
        raise_errors=raise_errors,
        verbose=verbose,
        token=token_str_bool,
        ignore_duplicates=ignore_duplicates,
        cache_dir=cache_dir,
        framework=framework,
        device=device,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        use_flash_attention=use_flash_attention,
        clear_model_cache=clear_model_cache,
        only_validation_split=only_validation_split,
        few_shot=few_shot,
    )

    # Perform the benchmark evaluation
    benchmarker(model_id=model_ids, dataset=datasets)


if __name__ == "__main__":
    benchmark()
