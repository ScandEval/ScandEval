"""Factory class for creating dataset configurations."""

import os

import torch

from scandeval.dataset_configs import get_all_dataset_configs
from scandeval.exceptions import InvalidBenchmark

from .config import BenchmarkConfig, Language, Task
from .enums import Device, Framework
from .languages import get_all_languages
from .tasks import get_all_tasks


def build_benchmark_config(
    progress_bar: bool,
    save_results: bool,
    task: str | list[str] | None,
    dataset: str | list[str] | None,
    language: str | list[str],
    model_language: str | list[str] | None,
    dataset_language: str | list[str] | None,
    framework: Framework | str | None,
    device: Device | None,
    batch_size: int,
    evaluate_train: bool,
    raise_errors: bool,
    cache_dir: str,
    token: bool | str,
    openai_api_key: str | None,
    force: bool,
    verbose: bool,
    trust_remote_code: bool,
    load_in_4bit: bool | None,
    use_flash_attention: bool,
    clear_model_cache: bool,
    only_validation_split: bool,
    few_shot: bool,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
        progress_bar:
            Whether to show a progress bar when running the benchmark.
        save_results:
            Whether to save the benchmark results to a file.
        task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.
        dataset:
            The datasets to include for task. If None then all datasets will be
            included, limited by the `task` parameter.
        language:
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages (also non-Scandinavian) should be considered.
        model_language:
            The language codes of the languages to include for models. If None then
            the `language` parameter will be used.
        dataset_language:
            The language codes of the languages to include for datasets. If None then
            the `language` parameter will be used.
        framework:
            The framework to use for running the models. If None then the framework
            will be set automatically.
        device:
            The device to use for running the models. If None then the device will be
            set automatically.
        batch_size:
            The batch size to use for running the models.
        evaluate_train:
            Whether to evaluate the models on the training set.
        raise_errors:
            Whether to raise errors when running the benchmark.
        cache_dir:
            The directory to use for caching the models.
        token:
            The token to use for running the models.
        openai_api_key:
            The OpenAI API key to use for running the models.
        force:
            Whether to force the benchmark to run even if the results are already
            cached.
        verbose:
            Whether to print verbose output when running the benchmark.
        trust_remote_code:
            Whether to trust remote code when running the benchmark.
        load_in_4bit:
            Whether to load the models in 4-bit precision.
        use_flash_attention:
            Whether to use Flash Attention for the models.
        clear_model_cache:
            Whether to clear the model cache before running the benchmark.
        only_validation_split:
            Whether to only use the validation split for the datasets.
        few_shot:
            Whether to use few-shot learning for the models.

    Returns:
        The benchmark configuration.
    """
    language_codes = get_correct_language_codes(language_codes=language)
    model_languages = prepare_languages(
        language_codes=model_language, default_language_codes=language_codes
    )
    dataset_languages = prepare_languages(
        language_codes=dataset_language, default_language_codes=language_codes
    )

    tasks, datasets = prepare_tasks_and_datasets(task=task, dataset=dataset)

    torch_device = prepare_device(device=device)

    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    framework_obj = Framework(framework) if framework is not None else None

    return BenchmarkConfig(
        model_languages=model_languages,
        dataset_languages=dataset_languages,
        tasks=tasks,
        datasets=datasets,
        batch_size=batch_size,
        raise_errors=raise_errors,
        cache_dir=cache_dir,
        evaluate_train=evaluate_train,
        token=token,
        openai_api_key=openai_api_key,
        force=force,
        progress_bar=progress_bar,
        save_results=save_results,
        verbose=verbose,
        framework=framework_obj,
        device=torch_device,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        use_flash_attention=use_flash_attention,
        clear_model_cache=clear_model_cache,
        only_validation_split=only_validation_split,
        few_shot=few_shot,
    )


def get_correct_language_codes(language_codes: str | list[str]) -> list[str]:
    """Get correct language-code(s).

    Args:
        language_codes:
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages (also non-Scandinavian) should be considered.

    Returns:
        The correct language-codes.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `languages`
    if "all" in language_codes:
        languages = list(language_mapping.keys())
    elif isinstance(language_codes, str):
        languages = [language_codes]
    else:
        languages = language_codes

    # If `languages` contains 'no' then also include 'nb' and 'nn'. Conversely, if
    # either 'nb' or 'nn' are specified then also include 'no'.
    if "no" in languages:
        languages = list(set(languages) | {"nb", "nn"})
    elif "nb" in languages or "nn" in languages:
        languages = list(set(languages) | {"no"})

    return languages


def prepare_languages(
    language_codes: str | list[str] | None, default_language_codes: list[str]
) -> list[Language]:
    """Prepare language(s) for benchmarking.

    Args:
        language_codes:
            The language codes of the languages to include for models or datasets.
            If specified then this overrides the `language` parameter for model or
            dataset languages.
        default_language_codes:
            The default language codes of the languages to include.

    Returns:
        The prepared model or dataset languages.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `languages_str` of language codes to use for models or datasets
    languages_str: list[str]
    if language_codes is None:
        languages_str = default_language_codes
    elif isinstance(language_codes, str):
        languages_str = [language_codes]
    else:
        languages_str = language_codes

    # Convert the model languages to language objects
    if "all" in languages_str:
        prepared_languages = list(language_mapping.values())
    else:
        prepared_languages = [language_mapping[language] for language in languages_str]

    return prepared_languages


def prepare_tasks_and_datasets(
    task: str | list[str] | None, dataset: str | list[str] | None
) -> tuple[list[Task], list[str]]:
    """Prepare task(s) and dataset(s) for benchmarking.

    Args:
        task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.
        dataset:
            The datasets to include for task. If None then all datasets will be
            included, limited by the `task` parameter.

    Returns:
        The prepared tasks and datasets.

    Raises:
        InvalidBenchmark:
            If the task or dataset is not found in the benchmark tasks or datasets.
    """
    # Create a dictionary that maps benchmark tasks to their associated benchmark
    # task objects, and a dictionary that maps dataset names to their associated
    # dataset configuration objects
    task_mapping = get_all_tasks()
    all_dataset_configs = get_all_dataset_configs()

    # Create the list of dataset tasks
    try:
        if task is None:
            tasks = list(task_mapping.values())
        elif isinstance(task, str):
            tasks = [task_mapping[task]]
        else:
            tasks = [task_mapping[task] for task in task]
    except KeyError as e:
        raise InvalidBenchmark(f"Task {e} not found in the benchmark tasks.") from e

    all_datasets = list(all_dataset_configs.keys())
    if dataset is None:
        dataset = all_datasets
    elif isinstance(dataset, str):
        dataset = [dataset]

    invalid_datasets = set(dataset) - set(all_datasets)
    if invalid_datasets:
        raise InvalidBenchmark(
            f"Dataset(s) {', '.join(invalid_datasets)} not found in the benchmark "
            "datasets."
        )

    # Create the list of dataset names
    datasets = [
        dataset_name
        for dataset_name, dataset_config in all_dataset_configs.items()
        if dataset_name in dataset and dataset_config.task in tasks
    ]

    return tasks, datasets


def prepare_device(device: Device | None) -> torch.device:
    """Prepare device for benchmarking.

    Args:
        device:
            The device to use for running the models. If None then the device will be
            set automatically.

    Returns:
        The prepared device.
    """
    device_mapping = {
        Device.CPU: torch.device("cpu"),
        Device.CUDA: torch.device("cuda"),
        Device.MPS: torch.device("mps"),
    }
    if isinstance(device, Device):
        return device_mapping[device]

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
