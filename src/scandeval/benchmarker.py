"""Class that benchmarks Scandinavian language models."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .config import BenchmarkConfig, DatasetConfig
from .dataset_factory import DatasetFactory
from .datasets import get_all_dataset_configs
from .exceptions import InvalidBenchmark
from .hf_hub import get_model_lists

logger = logging.getLogger(__name__)


class Benchmarker:
    """Benchmarking all the Scandinavian language models.

    Args:
        progress_bar (bool, optional):
            Whether progress bars should be shown. Defaults to True.
        save_results (bool, optional):
            Whether to save the benchmark results to
            'scandeval_benchmark_results.json'. Defaults to False.
        language (str or list of str, optional):
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this to
            'all' if all languages (also non-Scandinavian) should be considered.
            Defaults to ['da', 'sv', 'no'].
        model_language (str or list of str or None, optional):
            The language codes of the languages to include for models. If specified
            then this overrides the `language` parameter for model languages. Defaults
            to None.
        dataset_language (str or list of str or None, optional):
            The language codes of the languages to include for datasets. If specified
            then this overrides the `language` parameter for dataset languages. Defaults
            to None.
        task (str or list of str, optional):
            The tasks to consider in the list. Set this to 'all' if all
            tasks should be considered. Defaults to 'all'.
        evaluate_train (bool, optional):
            Whether to evaluate the training set as well. Defaults to False.
        raise_error_on_invalid_model (bool, optional):
            Whether to raise an error if a model is invalid. Defaults to False.
        cache_dir (str, optional):
            Directory to store cached models. Defaults to '.scandeval_cache'.
        use_auth_token (bool, optional):
            Whether the benchmark should use an authentication token. Defaults
            to False.
        verbose (bool, optional):
            Whether to output additional output. Defaults to False.

    Attributes:
        progress_bar (bool): Whether progress bars should be shown.
        save_results (bool): Whether to save the benchmark results.
        language (str or list of str): The languages to include in the list.
        task (str or list of str): The tasks to consider in the list.
        evaluate_train (bool): Whether to evaluate the training set as well.
        verbose (bool): Whether to output additional output.
        use_auth_token (bool): Whether an authentication token should be used.
        benchmark_results (dict): The benchmark results.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = False,
        language: Union[str, Sequence] = ["da", "sv", "no"],
        model_language: Optional[Union[str, Sequence]] = None,
        dataset_language: Optional[Union[str, Sequence]] = None,
        task: Union[str, Sequence] = "all",
        evaluate_train: bool = False,
        raise_error_on_invalid_model: bool = False,
        cache_dir: str = ".scandeval_cache",
        use_auth_token: bool = False,
        verbose: bool = False,
    ):
        # Create the list `languages`
        languages: Sequence[Optional[str]]
        if language == "all":
            languages = [None]
        elif isinstance(language, str):
            languages = [language]
        else:
            languages = language

        # Create the list `model_languages`
        model_languages: Sequence[Optional[str]]
        if model_language == "all":
            model_languages = [None]
        elif isinstance(model_language, str):
            model_languages = [model_language]
        else:
            model_languages = languages

        # Create the list `dataset_languages`
        dataset_languages: Sequence[Optional[str]]
        if dataset_language == "all":
            dataset_languages = [None]
        elif isinstance(dataset_language, str):
            dataset_languages = [dataset_language]
        else:
            dataset_languages = languages

        # Create the list `tasks`
        tasks: Sequence[Optional[str]]
        if task == "all":
            tasks = [None]
        elif isinstance(task, str):
            tasks = [task]
        else:
            tasks = task

        # If `languages` contains 'no' then also include 'nb' and 'nn'. Conversely, if
        # either 'nb' or 'nn' are specified then also include 'no'.
        if "no" in languages:
            languages = list(set(languages) | {"nb", "nn"})
        elif "nb" in languages or "nn" in languages:
            languages = list(set(languages) | {"no"})

        # Build benchmark config and store it
        self.benchmark_config = BenchmarkConfig(
            languages=languages,
            model_languages=model_languages,
            dataset_languages=dataset_languages,
            tasks=tasks,
            raise_error_on_invalid_model=raise_error_on_invalid_model,
            cache_dir=cache_dir,
            evaluate_train=evaluate_train,
            use_auth_token=use_auth_token,
            progress_bar=progress_bar,
            save_results=save_results,
            verbose=verbose,
        )

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: Union[Dict[str, Sequence[str]], None] = None

        # Initialise variable storing all benchmark results, which will be
        # updated as more models are benchmarked
        self.benchmark_results: Dict[str, dict] = defaultdict(dict)

        # Set logging level based on verbosity
        logging_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(logging_level)

        # Initialise a dataset factory
        self.dataset_factory = DatasetFactory(benchmark_config=self.benchmark_config)

    def benchmark(
        self,
        model_id: Optional[Union[Sequence[str], str]] = None,
        dataset: Optional[Union[Sequence[str], str]] = None,
    ) -> Dict[str, Dict[str, dict]]:
        """Benchmarks models on datasets.

        Args:
            model_id (str, list of str or None, optional):
                The model ID(s) of the models to benchmark. If None then all
                relevant model IDs will be benchmarked. Defaults to None.
            dataset (str, list of str or None, optional):
                The datasets to benchmark on. If None then all datasets will
                be benchmarked. Defaults to None.

        Returns:
            dict:
                A nested dictionary of the benchmark results. The keys are the
                names of the datasets, with values being new dictionaries
                having the model IDs as keys.
        """
        # If `model_id` is not specified, then fetch all the relevant model IDs
        model_ids: Sequence[str]
        if model_id is None:
            model_ids = self._get_fresh_model_ids(
                model_languages=self.benchmark_config.model_languages,
                tasks=self.benchmark_config.tasks,
            )
        elif isinstance(model_id, str):
            model_ids = [model_id]
        else:
            model_ids = model_id

        # Get all the relevant dataset configurations
        if dataset is None:
            dataset_configs = [
                cfg
                for cfg in get_all_dataset_configs()
                if any(
                    lang in self.benchmark_config.dataset_languages
                    for lang in cfg.languages
                )
            ]
        elif isinstance(dataset, str):
            dataset_configs = [
                cfg for cfg in get_all_dataset_configs() if cfg.name == dataset
            ]
        else:
            dataset_configs = [
                cfg for cfg in get_all_dataset_configs() if cfg.name in dataset
            ]

        # Benchmark all the models in `model_ids` on all the datasets in
        # `benchmarks`
        for dataset_config in dataset_configs:
            for m_id in model_ids:
                self._benchmark_single(
                    dataset_config=dataset_config,
                    model_id=m_id,
                )

        # Save the benchmark results
        if self.benchmark_config.save_results:
            output_path = Path.cwd() / "scandeval_benchmark_results.json"
            with output_path.open("w") as f:
                json.dump(self.benchmark_results, f)

        return self.benchmark_results

    def _benchmark_single(
        self,
        dataset_config: DatasetConfig,
        model_id: str,
    ):
        """Benchmark a single model on a single dataset.

        Args:
            dataset_config (DatasetConfig):
                The dataset configuration to use.
            model_id (str):
                The model ID to use.
        """
        logger.info(f"Benchmarking {model_id} on {dataset_config.pretty_name}")
        try:
            dataset_obj = self.dataset_factory.build_dataset(dataset_config)
            results = dataset_obj(model_id)
            self.benchmark_results[dataset_config.name][model_id] = results
            logger.debug(f"Results:\n{results}")

        except InvalidBenchmark as e:

            # If the model ID is not valid then raise an error, if
            # specified
            model_err_msg = "does not exist on the Hugging Face Hub"
            if (
                self.benchmark_config.raise_error_on_invalid_model
                and model_err_msg in str(e)
            ):
                raise e

            # Otherwise, log the error
            else:
                logger.info(
                    f"{model_id} could not be benchmarked on "
                    f"{dataset_config.pretty_name}. Skipping."
                )
                logger.debug(f'The error message was "{e}".')

    def __call__(self, *args, **kwargs):
        return self.benchmark(*args, **kwargs)

    def _get_fresh_model_ids(
        self,
        model_languages: Sequence[Optional[str]],
        tasks: Sequence[Optional[str]],
    ) -> list:
        """Get list of model IDs from the Hugging Face Hub.

        Returns:
            list: Sequence of model IDs.
        """
        # If the model lists have not been fetched already, then do it
        if self._model_lists is None:
            logger.info("Fetching list of models from the Hugging Face Hub")
            self._model_lists = get_model_lists(
                languages=model_languages,
                tasks=tasks,
                use_auth_token=self.benchmark_config.use_auth_token,
            )
        try:
            model_ids: List[str] = list()
            for language in model_languages:
                if language is not None:
                    model_ids.extend(self._model_lists[language])  # type: ignore
            for task in tasks:
                if task is not None:
                    model_ids.extend(self._model_lists[task])  # type: ignore
            model_ids.extend(self._model_lists["multilingual"])  # type: ignore

        # If the model list corresponding to the language or task was not present
        # in the stored model lists, then fetch new model lists and try again
        except KeyError:
            self._model_lists = get_model_lists(
                languages=model_languages,
                tasks=tasks,
                use_auth_token=self.benchmark_config.use_auth_token,
            )
            model_ids = list()
            for language in model_languages:
                if language is not None:
                    model_ids.extend(self._model_lists[language])  # type: ignore
            for task in tasks:
                if task is not None:
                    model_ids.extend(self._model_lists[task])  # type: ignore
            model_ids.extend(self._model_lists["multilingual"])  # type: ignore

        # Remove duplicate model IDs
        model_ids = list(set(model_ids))

        return model_ids
