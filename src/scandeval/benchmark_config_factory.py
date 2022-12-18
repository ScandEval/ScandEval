"""Factory class for creating dataset configurations."""

from typing import List, Optional, Sequence, Union

from .config import BenchmarkConfig, DatasetTask, Language
from .dataset_tasks import get_all_dataset_tasks
from .languages import get_all_languages


def build_benchmark_config(
    language: Union[str, List[str]],
    model_language: Optional[Union[str, Sequence[str]]],
    dataset_language: Optional[Union[str, Sequence[str]]],
    dataset_task: Optional[Union[str, Sequence[str]]],
    batch_size: int,
    raise_error_on_invalid_model: bool,
    cache_dir: str,
    evaluate_train: bool,
    use_auth_token: Union[bool, str],
    progress_bar: bool,
    save_results: bool,
    verbose: bool,
) -> BenchmarkConfig:
    """Create a benchmark configuration.

    Args:
        language (str or list of str, optional):
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages (also non-Scandinavian) should be considered.
        model_language (None, str or sequence of str, optional):
            The language codes of the languages to include for models. If specified
            then this overrides the `language` parameter for model languages.
        dataset_language (None, str or sequence of str, optional):
            The language codes of the languages to include for datasets. If
            specified then this overrides the `language` parameter for dataset
            languages.
        dataset_task (str or sequence of str, optional):
            The tasks to include for dataset. If "all" then datasets will not be
            filtered based on their task.
        batch_size (int):
            The batch size to use.
        raise_error_on_invalid_model (bool, optional):
            Whether to raise an error if a model is invalid.
        cache_dir (str, optional):
            Directory to store cached models.
        evaluate_train (bool, optional):
            Whether to evaluate the training set as well.
        use_auth_token (bool or str, optional):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        progress_bar (bool, optional):
            Whether progress bars should be shown.
        save_results (bool, optional):
            Whether to save the benchmark results to
            'scandeval_benchmark_results.json'.
        verbose (bool, optional):
            Whether to output additional output.
    """
    # Prepare the languages
    languages = prepare_languages(language=language)

    # Prepare the model languages
    model_languages = prepare_model_languages(
        model_language=model_language,
        languages=languages,
    )

    # Prepare the dataset languages
    dataset_languages = prepare_dataset_languages(
        dataset_language=dataset_language,
        languages=languages,
    )

    # Prepare the dataset tasks
    dataset_tasks = prepare_dataset_tasks(dataset_task=dataset_task)

    # Build benchmark config and return it
    return BenchmarkConfig(
        model_languages=model_languages,
        dataset_languages=dataset_languages,
        dataset_tasks=dataset_tasks,
        batch_size=batch_size,
        raise_error_on_invalid_model=raise_error_on_invalid_model,
        cache_dir=cache_dir,
        evaluate_train=evaluate_train,
        use_auth_token=use_auth_token,
        progress_bar=progress_bar,
        save_results=save_results,
        verbose=verbose,
    )


def prepare_languages(language: Union[str, List[str]]) -> Sequence[str]:
    """Prepare language(s) for benchmarking.

    Args:
        language (str or list of str):
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages (also non-Scandinavian) should be considered.

    Returns:
        sequence of str:
            The prepared languages.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `languages`
    if "all" in language:
        languages = list(language_mapping.keys())
    elif isinstance(language, str):
        languages = [language]
    else:
        languages = language

    # If `languages` contains 'no' then also include 'nb' and 'nn'. Conversely, if
    # either 'nb' or 'nn' are specified then also include 'no'.
    if "no" in languages:
        languages = list(set(languages) | {"nb", "nn"})
    elif "nb" in languages or "nn" in languages:
        languages = list(set(languages) | {"no"})

    return languages


def prepare_model_languages(
    model_language: Optional[Union[str, Sequence[str]]],
    languages: Sequence[str],
) -> Sequence[Language]:
    """Prepare model language(s) for benchmarking.

    Args:
        model_language (None, str or sequence of str):
            The language codes of the languages to include for models. If specified
            then this overrides the `language` parameter for model languages.
        languages (sequence of str):
            The default language codes of the languages to include.

    Returns:
        sequence of Language objects:
            The prepared model languages.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `model_languages`
    model_languages_str: Sequence[str]
    if model_language is None:
        model_languages_str = languages
    elif isinstance(model_language, str):
        model_languages_str = [model_language]
    else:
        model_languages_str = model_language

    # Convert the model languages to language objects
    if "all" in model_languages_str:
        model_languages = list(language_mapping.values())
    else:
        model_languages = [
            language_mapping[language] for language in model_languages_str
        ]

    return model_languages


def prepare_dataset_languages(
    dataset_language: Optional[Union[str, Sequence[str]]],
    languages: Sequence[str],
) -> Sequence[Language]:
    """Prepare dataset language(s) for benchmarking.

    Args:
        model_language (None, str or sequence of str):
            The language codes of the languages to include for datasets. If
            specified then this overrides the `language` parameter for dataset
            languages.
        languages (sequence of str):
            The default language codes of the languages to include.

    Returns:
        sequence of Language objects:
            The prepared dataset languages.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `dataset_languages_str`
    dataset_languages_str: Sequence[str]
    if dataset_language is None:
        dataset_languages_str = languages
    elif isinstance(dataset_language, str):
        dataset_languages_str = [dataset_language]
    else:
        dataset_languages_str = dataset_language

    # Convert the dataset languages to language objects
    if "all" in dataset_languages_str:
        dataset_languages = list(language_mapping.values())
    else:
        dataset_languages = [
            language_mapping[language] for language in dataset_languages_str
        ]

    return dataset_languages


def prepare_dataset_tasks(
    dataset_task: Optional[Union[str, Sequence[str]]]
) -> Sequence[DatasetTask]:
    """Prepare dataset task(s) for benchmarking.

    Args:
        dataset_task (str or sequence of str, optional):
            The tasks to include for dataset. If "all" then datasets will not be
            filtered based on their task. Defaults to "all".

    Returns:
        sequence of DatasetTask:
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
