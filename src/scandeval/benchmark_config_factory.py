"""Factory class for creating dataset configurations."""

import torch

from .config import BenchmarkConfig, DatasetTask, Language
from .dataset_tasks import get_all_dataset_tasks
from .enums import Device, Framework
from .languages import get_all_languages


def build_benchmark_config(
    language: str | list[str],
    model_language: str | list[str] | None,
    dataset_language: str | list[str] | None,
    dataset_task: str | list[str] | None,
    batch_size: int,
    raise_errors: bool,
    cache_dir: str,
    evaluate_train: bool,
    token: bool | str,
    openai_api_key: str | None,
    progress_bar: bool,
    save_results: bool,
    verbose: bool,
    framework: Framework | str | None,
    device: Device | None,
    trust_remote_code: bool,
    load_in_4bit: bool | None,
    use_flash_attention: bool,
    clear_model_cache: bool,
    only_validation_split: bool,
    few_shot: bool,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
        language:
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages (also non-Scandinavian) should be considered.
        model_language:
            The language codes of the languages to include for models. If specified
            then this overrides the `language` parameter for model languages.
        dataset_language:
            The language codes of the languages to include for datasets. If
            specified then this overrides the `language` parameter for dataset
            languages.
        dataset_task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.
        batch_size:
            The batch size to use.
        raise_errors:
            Whether to raise errore instead of skipping them.
        cache_dir:
            Directory to store cached models.
        evaluate_train:
            Whether to evaluate the training set as well.
        token:
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.
        openai_api_key:
            The OpenAI API key to use for authentication. If None, then None will be
            returned.
        progress_bar:
            Whether progress bars should be shown.
        save_results:
            Whether to save the benchmark results to local JSON file.
        verbose:
            Whether to output additional output.
        framework:
            The model framework to use. If None then the framework will be set
            automatically. Only relevant if `model_id` refers to a local model.
        device:
            The device to use for running the models. If None then the device will be
            set automatically.
        trust_remote_code:
            Whether to trust remote code when loading models from the Hugging Face
            Hub.
        load_in_4bit:
            Whether to load models in 4-bit precision. If None then this will be done
            if CUDA is available and the model is a decoder model. Defaults to None.
        use_flash_attention:
            Whether to use Flash Attention.
        clear_model_cache:
            Whether to clear the model cache after benchmarking each model.
        only_validation_split:
            Whether to only evaluate on the validation split.
        few_shot:
            Whether to only evaluate the model using few-shot evaluation. Only relevant
            if the model is generative.
    """
    language_codes = get_correct_language_codes(language_codes=language)
    model_languages = prepare_languages(
        language_codes=model_language, default_language_codes=language_codes
    )
    dataset_languages = prepare_languages(
        language_codes=dataset_language, default_language_codes=language_codes
    )

    dataset_tasks = prepare_dataset_tasks(dataset_task=dataset_task)

    torch_device = prepare_device(device=device)

    return BenchmarkConfig(
        model_languages=model_languages,
        dataset_languages=dataset_languages,
        dataset_tasks=dataset_tasks,
        batch_size=batch_size,
        raise_errors=raise_errors,
        cache_dir=cache_dir,
        evaluate_train=evaluate_train,
        token=token,
        openai_api_key=openai_api_key,
        progress_bar=progress_bar,
        save_results=save_results,
        verbose=verbose,
        framework=framework,
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


def prepare_dataset_tasks(dataset_task: str | list[str] | None) -> list[DatasetTask]:
    """Prepare dataset task(s) for benchmarking.

    Args:
        dataset_task:
            The tasks to include for dataset. If None then datasets will not be
            filtered based on their task.

    Returns:
        The prepared dataset tasks.
    """
    # Create a dictionary that maps benchmark tasks to their associated benchmark
    # task objects
    dataset_task_mapping = get_all_dataset_tasks()

    # Create the list of dataset tasks
    if dataset_task is None:
        dataset_tasks = list(dataset_task_mapping.values())
    elif isinstance(dataset_task, str):
        dataset_tasks = [dataset_task_mapping[dataset_task]]
    else:
        dataset_tasks = [dataset_task_mapping[task] for task in dataset_task]

    return dataset_tasks


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
