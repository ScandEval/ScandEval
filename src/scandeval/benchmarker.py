"""Class that benchmarks Scandinavian language models."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .benchmark_config_factory import build_benchmark_config
from .config import DatasetConfig, Language
from .dataset_configs import get_all_dataset_configs
from .dataset_factory import DatasetFactory
from .exceptions import InvalidBenchmark
from .hf_hub import get_model_lists
from .types import SCORE_DICT

# Create logger
logger = logging.getLogger(__name__)


class Benchmarker:
    """Benchmarking all the Scandinavian language models.

    Args:
        progress_bar (bool, optional):
            Whether progress bars should be shown. Defaults to True.
        save_results (bool, optional):
            Whether to save the benchmark results to
            'scandeval_benchmark_results.jsonl'. Defaults to False.
        language (str or list of str, optional):
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this to
            'all' if all languages (also non-Scandinavian) should be considered.
            Defaults to ['da', 'sv', 'no'].
        model_language (None, str or sequence of str, optional):
            The language codes of the languages to include for models. If specified
            then this overrides the `language` parameter for model languages. Defaults
            to None.
        dataset_language (None, str or sequence of str, optional):
            The language codes of the languages to include for datasets. If specified
            then this overrides the `language` parameter for dataset languages.
            Defaults to None.
        dataset_task (str or sequence of str, optional):
            The tasks to include for dataset. If "all" then datasets will not be
            filtered based on their task. Defaults to "all".
        batch_size (int, optional):
            The batch size to use. Defaults to 32.
        evaluate_train (bool, optional):
            Whether to evaluate the training set as well. Defaults to False.
        raise_error_on_invalid_model (bool, optional):
            Whether to raise an error if a model is invalid. Defaults to False.
        cache_dir (str, optional):
            Directory to store cached models. Defaults to '.scandeval_cache'.
        use_auth_token (bool or str, optional):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        ignore_duplicates (bool, optional):
            Whether to skip evaluation of models which have already been evaluated,
            with scores lying in the 'scandeval_benchmark_results.jsonl' file. Defaults
            to True.
        verbose (bool, optional):
            Whether to output additional output. Defaults to False.

    Attributes:
        progress_bar (bool): Whether progress bars should be shown.
        save_results (bool): Whether to save the benchmark results.
        language (str or list of str): The languages to include in the list.
        dataset_task (str or list of str): The dataset tasks to include.
        evaluate_train (bool): Whether to evaluate the training set as well.
        verbose (bool): Whether to output additional output.
        use_auth_token (str or bool): The authentication token for the Hugging Face Hub.
        benchmark_results (dict): The benchmark results.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = False,
        language: Union[str, List[str]] = ["da", "sv", "no"],
        model_language: Optional[Union[str, Sequence[str]]] = None,
        dataset_language: Optional[Union[str, Sequence[str]]] = None,
        dataset_task: Optional[Union[str, Sequence[str]]] = None,
        batch_size: int = 32,
        evaluate_train: bool = False,
        raise_error_on_invalid_model: bool = False,
        cache_dir: str = ".scandeval_cache",
        use_auth_token: Union[bool, str] = False,
        ignore_duplicates: bool = True,
        verbose: bool = False,
    ) -> None:
        # Build benchmark configuration
        self.benchmark_config = build_benchmark_config(
            language=language,
            model_language=model_language,
            dataset_language=dataset_language,
            dataset_task=dataset_task,
            batch_size=batch_size,
            raise_error_on_invalid_model=raise_error_on_invalid_model,
            cache_dir=cache_dir,
            evaluate_train=evaluate_train,
            use_auth_token=use_auth_token,
            progress_bar=progress_bar,
            save_results=save_results,
            verbose=verbose,
        )

        # Set attributes from arguments
        self.ignore_duplicates = ignore_duplicates

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: Union[Dict[str, Sequence[str]], None] = None

        # Set up the results path
        self.results_path = Path.cwd() / "scandeval_benchmark_results.jsonl"

        # Set up the benchmark results variable, which will be populated with the
        # contents of the results file if it exists. If not, then it will be an empty
        # list
        self.benchmark_results: List[Dict[str, Union[str, int, List[str], SCORE_DICT]]]
        if self.results_path.exists():
            with self.results_path.open() as f:
                self.benchmark_results = [
                    json.loads(line) for line in f if line.strip()
                ]
        else:
            self.benchmark_results = list()

        # Set logging level based on verbosity
        logging_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(logging_level)

        # Initialise a dataset factory
        self.dataset_factory = DatasetFactory(benchmark_config=self.benchmark_config)

    def benchmark(
        self,
        model_id: Optional[Union[Sequence[str], str]] = None,
        dataset: Optional[Union[Sequence[str], str]] = None,
    ) -> List[Dict[str, Union[str, int, List[str], SCORE_DICT]]]:
        """Benchmarks models on datasets.

        Args:
            model_id (str, list of str or None, optional):
                The full Hugging Face Hub path(s) to the pretrained transformer model.
                The specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id,
                and defaults to the latest version if not specified. If None then all
                relevant model IDs will be benchmarked. Defaults to None.
            dataset (str, list of str or None, optional):
                The datasets to benchmark on. If None then all datasets will be
                benchmarked. Defaults to None.

        Returns:
            list of dict:
                The benchmark results, where each result is a dictionary with the keys
                'dataset', 'task', 'dataset_languages', 'model', 'num_model_parameters'
                and 'scores'. If an error occured then the dictionary will only contain
                the key 'error', with the associated value being the error message.
        """
        # Prepare the model IDs
        model_ids = self._prepare_model_ids(model_id)

        # Get all the relevant dataset configurations
        dataset_configs = self._prepare_dataset_configs(dataset)

        # Iterate over all the models and datasets
        for m_id in model_ids:
            for dataset_config in dataset_configs:

                # Skip if we have already benchmarked this model on this dataset and
                # `ignore_duplicates` is set
                if self.ignore_duplicates and self._has_been_benchmarked(
                    model_id=m_id, dataset=dataset_config.name
                ):
                    logger.debug(
                        f"Skipping benchmarking {m_id} on {dataset_config.pretty_name},"
                        " as it has already been benchmarked."
                    )
                    continue

                # Benchmark a single model on a single dataset
                record = self._benchmark_single(
                    dataset_config=dataset_config,
                    model_id=m_id,
                )

                # If the benchmark was unsuccessful then skip
                if "error" in record:
                    error_msg = record["error"]
                    logger.info(
                        f"{m_id} could not be benchmarked on "
                        f"{dataset_config.pretty_name}. Skipping."
                    )
                    logger.debug(f"The error message was {error_msg!r}.")
                    continue

                # Add the record to the benchmark results
                self.benchmark_results.append(record)

                # Save the benchmark results
                if self.benchmark_config.save_results:
                    with self.results_path.open("a") as f:
                        f.write("\n" + json.dumps(record))

        return self.benchmark_results

    def _has_been_benchmarked(self, model_id: str, dataset: str) -> bool:
        """Checks whether a model has already been benchmarked on a dataset.

        Args:
            model_id (str):
                The model ID.
            dataset (str):
                The dataset.

        Returns:
            bool:
                Whether the model has already been evaluated on the dataset.
        """
        for record in self.benchmark_results:
            if record["model"] == model_id and record["dataset"] == dataset:
                return True
        return False

    def _prepare_model_ids(
        self,
        model_id: Optional[Union[Sequence[str], str]],
    ) -> Sequence[str]:
        """Prepare the model ID(s) to be benchmarked.

        Args:
            model_id (str, list of str or None):
                The model ID(s) of the models to benchmark. If None then all model IDs
                will be retrieved.

        Returns:
            sequence of str:
                The prepared list of model IDs.
        """
        model_ids: Sequence[str]

        # If `model_id` is not specified, then fetch all the relevant model IDs
        if model_id is None:
            model_ids = self._get_model_ids(
                languages=self.benchmark_config.model_languages,
            )

        # Otherwise, if `model_id` is a string, ensure that it is a list
        elif isinstance(model_id, str):
            model_ids = [model_id]

        # Otherwise `model_id` is already a list, so we do nothing
        else:
            model_ids = model_id

        # Reorder the `model_ids` list to include the ones present in the benchmark
        # results first
        benchmarked_model_ids = [record["model"] for record in self.benchmark_results]
        model_ids_sorted = [m_id for m_id in model_ids if m_id in benchmarked_model_ids]
        model_ids_sorted += [
            m_id for m_id in model_ids if m_id not in benchmarked_model_ids
        ]

        return model_ids_sorted

    def _prepare_dataset_configs(
        self,
        dataset: Optional[Union[Sequence[str], str]],
    ) -> Sequence[DatasetConfig]:
        """Prepare the dataset configuration(s) to be benchmarked.

        Args:
            dataset (str, list of str or None, optional):
                The datasets to benchmark on. If None then all datasets will be
                benchmarked. Defaults to None.

        Returns:
            sequence of str:
                The prepared list of model IDs.
        """
        if dataset is None:
            dataset_configs = [
                cfg
                for cfg in get_all_dataset_configs().values()
                if any(
                    lang in self.benchmark_config.dataset_languages
                    for lang in cfg.languages
                )
                and cfg.task in self.benchmark_config.dataset_tasks
            ]
        elif isinstance(dataset, str):
            dataset_configs = [
                cfg for cfg in get_all_dataset_configs().values() if cfg.name == dataset
            ]
        else:
            dataset_configs = [
                cfg for cfg in get_all_dataset_configs().values() if cfg.name in dataset
            ]

        return dataset_configs

    def _benchmark_single(
        self,
        dataset_config: DatasetConfig,
        model_id: str,
    ) -> Dict[str, Union[str, int, List[str], SCORE_DICT]]:
        """Benchmark a single model on a single dataset.

        Args:
            dataset_config (DatasetConfig):
                The dataset configuration to use.
            model_id (str):
                The model ID to use.

        Returns:
            dict:
                The benchmark results, being a dictionary with the keys 'dataset',
                'task', 'dataset_languages', 'model', 'num_model_parameters' and
                'scores'. If an error occured then the dictionary will only contain the
                key 'error', with the associated value being the error message.
        """
        logger.info(f"Benchmarking {model_id} on {dataset_config.pretty_name}")
        try:
            dataset = self.dataset_factory.build_dataset(dataset_config)
            results, metadata_dict = dataset(model_id)
            record: Dict[str, Union[str, int, List[str], SCORE_DICT]] = dict(
                dataset=dataset_config.name,
                task=dataset_config.task.name,
                dataset_languages=[
                    language.code for language in dataset_config.languages
                ],
                model=model_id,
                results=results,
                **metadata_dict,
            )
            logger.debug(f"Results:\n{results}")
            return record

        except InvalidBenchmark as e:

            # If the model ID is not valid then raise an error, if specified
            model_err_msg = "does not exist on the Hugging Face Hub"
            if (
                self.benchmark_config.raise_error_on_invalid_model
                and model_err_msg in str(e)
            ):
                raise e

            # Otherwise, if the error is due to the MPS fallback not being enabled,
            # then raise an error asking the user to enable it
            elif "PYTORCH_ENABLE_MPS_FALLBACK" in str(e):
                raise RuntimeError(
                    "The benchmark failed because the environment variable "
                    "`PYTORCH_ENABLE_MPS_FALLBACK` is not set. Please set this "
                    "environment variable to `1` and try again."
                )

            # Otherwise, return the error message
            else:
                return dict(error=str(e))

    def __call__(
        self, *args, **kwargs
    ) -> List[Dict[str, Union[str, int, List[str], SCORE_DICT]]]:
        return self.benchmark(*args, **kwargs)

    def _get_model_ids(
        self,
        languages: Sequence[Language],
    ) -> List[str]:
        """Get list of model IDs from the Hugging Face Hub.

        Args:
            languages (sequence of Language objects):
                The languages of the models to fetch.

        Returns:
            list of str:
                List of model IDs.
        """
        # Specify boolean variables determining whether the input variables are new
        new_languages = self._model_lists is not None and any(
            lang.code not in self._model_lists for lang in languages
        )

        # If the model lists have not been fetched already, then do it
        if self._model_lists is None or new_languages:
            self._model_lists = get_model_lists(
                languages=languages,
                use_auth_token=self.benchmark_config.use_auth_token,
            )

        # Extract all the model IDs from the model lists, for the chosen languages
        model_ids: List[str] = list()
        for language in languages:
            model_ids.extend(self._model_lists[language.code])

        # Add the multilingual models
        model_ids.extend(self._model_lists["multilingual"])

        # Add the fresh models
        model_ids.extend(self._model_lists["fresh"])

        # Remove duplicate model IDs
        model_ids = list(set(model_ids))

        return model_ids
