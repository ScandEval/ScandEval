"""Class that benchmarks Scandinavian language models."""

import importlib.metadata
import json
import logging
import re
import sys
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from .benchmark_config_factory import build_benchmark_config
from .config import BenchmarkConfig
from .dataset_configs import get_all_dataset_configs
from .dataset_factory import DatasetFactory
from .enums import Device, Framework
from .exceptions import InvalidBenchmark, InvalidModel
from .types import ScoreDict
from .utils import get_huggingface_model_lists

if TYPE_CHECKING:
    from .config import DatasetConfig, Language
    from .protocols import GenerativeModel, Tokenizer


logger = logging.getLogger(__package__)


class BenchmarkConfigParams(BaseModel):
    """The parameters for the benchmark configuration."""

    model_config = ConfigDict(protected_namespaces=())

    progress_bar: bool
    save_results: bool
    task: str | list[str] | None
    dataset: str | list[str] | None
    language: str | list[str]
    model_language: str | list[str] | None
    dataset_language: str | list[str] | None
    framework: Framework | str | None
    device: Device | None
    batch_size: int
    evaluate_train: bool
    raise_errors: bool
    cache_dir: str
    token: bool | str
    openai_api_key: str | None
    prefer_azure: bool
    azure_openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_version: str | None
    force: bool
    verbose: bool
    trust_remote_code: bool
    load_in_4bit: bool | None
    use_flash_attention: bool | None
    clear_model_cache: bool
    only_validation_split: bool
    few_shot: bool
    num_iterations: int
    run_with_cli: bool


class BenchmarkResult(BaseModel):
    """A benchmark result."""

    dataset: str
    task: str
    dataset_languages: list[str]
    model: str
    results: ScoreDict
    num_model_parameters: int
    max_sequence_length: int
    vocabulary_size: int
    generative: bool
    few_shot: bool
    validation_split: bool
    scandeval_version: str = importlib.metadata.version(__package__)

    @classmethod
    def from_dict(cls, config: dict) -> "BenchmarkResult":
        """Create a benchmark result from a dictionary.

        Args:
            config:
                The configuration dictionary.

        Returns:
            The benchmark result.
        """
        # To be backwards compatible, we accept old results which changed the model
        # name with parameters rather than adding them as explicit parameters
        val_matches = re.search(r"\(.*val.*\)$", config["model"])
        few_shot_matches = re.search(r"\(.*few-shot.*\)$", config["model"])
        config["model"] = re.sub(
            r"\(.*(few-shot|val).*\)$", "", config["model"]
        ).strip()

        # The default value for `few_shot` is True. It won't do anything if the model
        # is not generative, so this is fine
        if "generative" not in config:
            config["generative"] = few_shot_matches is not None
        if "few_shot" not in config:
            config["few_shot"] = True

        if "validation_split" not in config:
            config["validation_split"] = val_matches is not None

        return cls(**config)

    def append_to_results(self, results_path: Path) -> None:
        """Append the benchmark result to the results file.

        Args:
            results_path:
                The path to the results file.
        """
        json_str = json.dumps(self.model_dump())
        with results_path.open("a") as f:
            f.write("\n" + json_str)


class Benchmarker:
    """Benchmarking all the Scandinavian language models.

    Attributes:
        benchmark_config_default_params:
            The default parameters for the benchmark configuration.
        benchmark_config:
            The benchmark configuration.
        force:
            Whether to force evaluations of models, even if they have been benchmarked
            already.
        dataset_factory:
            The factory for creating datasets.
        results_path:
            The path to the results file.
        benchmark_results:
            The benchmark results.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = True,
        task: str | list[str] | None = None,
        dataset: list[str] | str | None = None,
        language: str | list[str] = "all",
        model_language: str | list[str] | None = None,
        dataset_language: str | list[str] | None = None,
        framework: Framework | str | None = None,
        device: Device | None = None,
        batch_size: int = 32,
        evaluate_train: bool = False,
        raise_errors: bool = False,
        cache_dir: str = ".scandeval_cache",
        token: bool | str = True,
        openai_api_key: str | None = None,
        prefer_azure: bool = False,
        azure_openai_api_key: str | None = None,
        azure_openai_endpoint: str | None = None,
        azure_openai_api_version: str | None = None,
        force: bool = False,
        verbose: bool = False,
        trust_remote_code: bool = False,
        load_in_4bit: bool | None = None,
        use_flash_attention: bool | None = None,
        clear_model_cache: bool = False,
        only_validation_split: bool = False,
        few_shot: bool = True,
        num_iterations: int = 10,
        run_with_cli: bool = False,
    ) -> None:
        """Initialise the benchmarker.

        Args:
            progress_bar:
                Whether progress bars should be shown. Defaults to True.
            save_results:
                Whether to save the benchmark results to
                'scandeval_benchmark_results.jsonl'. Defaults to True.
            task:
                The tasks benchmark the model(s) on. Mutually exclusive with `dataset`.
                If both `task` and `dataset` are None then all datasets will be
                benchmarked.
            dataset:
                The datasets to benchmark on. Mutually exclusive with `task`. If both
                `task` and `dataset` are None then all datasets will be benchmarked.
            language:
                The language codes of the languages to include, both for models and
                datasets. Set this to 'all' if all languages should be considered.
                Defaults to "all".
            model_language:
                The language codes of the languages to include for models. If specified
                then this overrides the `language` parameter for model languages.
                Defaults to None.
            dataset_language:
                The language codes of the languages to include for datasets. If
                specified then this overrides the `language` parameter for dataset
                languages. Defaults to None.
            framework:
                The model framework to use. Only relevant if `model-id` refers to a
                local path. Otherwise, the framework will be set automatically.
                Defaults to None.
            device:
                The device to use for benchmarking. Defaults to None.
            batch_size:
                The batch size to use. Defaults to 32.
            evaluate_train:
                Whether to evaluate the training set as well. Defaults to False.
            raise_errors:
                Whether to raise errors instead of skipping the model evaluation.
                Defaults to False.
            cache_dir:
                Directory to store cached models. Defaults to '.scandeval_cache'.
            token:
                The authentication token for the Hugging Face Hub. If a boolean value
                is specified then the token will be fetched from the Hugging Face CLI,
                where the user has logged in through `huggingface-cli login`. If a
                string is specified then it will be used as the token. Defaults to
                True.
            openai_api_key:
                The OpenAI API key to use for authentication. If None, then this will
                be loaded from the environment variable `OPENAI_API_KEY`. Defaults to
                None.
            prefer_azure:
                In the case where both OpenAI and Azure OpenAI models are available,
                whether to use the Azure OpenAI models. Defaults to False.
            azure_openai_api_key:
                The Azure OpenAI API key to use for authentication. If None, then this
                will be loaded from the environment variable `AZURE_OPENAI_API_KEY`.
                Defaults to None.
            azure_openai_endpoint:
                The Azure OpenAI endpoint to use for authentication. If None, then this
                will be loaded from the environment variable `AZURE_OPENAI_ENDPOINT`.
                Defaults to None.
            azure_openai_api_version:
                The Azure OpenAI API version to use for authentication. If None, then this
                will be loaded from the environment variable `AZURE_OPENAI_API_VERSION`.
                Defaults to None.
            force:
                Whether to force evaluations of models, even if they have been
                benchmarked already. Defaults to False.
            verbose:
                Whether to output additional output. Defaults to False.
            trust_remote_code:
                Whether to trust remote code when loading models. Defaults to False.
            load_in_4bit:
                Whether to load models in 4-bit precision. If None then this will be
                done if CUDA is available and the model is a decoder model. Defaults to
                None.
            use_flash_attention:
                Whether to use Flash Attention. If None then it will be used if it is
                installed and the model is a decoder model. Defaults to None.
            clear_model_cache:
                Whether to clear the model cache after benchmarking each model.
                Defaults to False.
            only_validation_split:
                Whether to only evaluate the validation split of the datasets. Defaults
                to False.
            few_shot:
                Whether to only evaluate the model using few-shot evaluation. Only
                relevant if the model is generative. Defaults to True.
            num_iterations:
                The number of times each model should be evaluated. This is only meant
                to be used for power users, and scores will not be allowed on the
                leaderboards if this is changed. Defaults to 10.
            run_with_cli:
                Whether the benchmarker is being run from the command-line interface.
                Defaults to False.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified.
        """
        if task is not None and dataset is not None:
            raise ValueError("Only one of `task` and `dataset` can be specified.")

        self.benchmark_config_default_params = BenchmarkConfigParams(
            progress_bar=progress_bar,
            save_results=save_results,
            task=task,
            dataset=dataset,
            language=language,
            model_language=model_language,
            dataset_language=dataset_language,
            framework=framework,
            device=device,
            batch_size=batch_size,
            evaluate_train=evaluate_train,
            raise_errors=raise_errors,
            cache_dir=cache_dir,
            token=token,
            openai_api_key=openai_api_key,
            prefer_azure=prefer_azure,
            azure_openai_api_key=azure_openai_api_key,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_api_version=azure_openai_api_version,
            force=force,
            verbose=verbose,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            use_flash_attention=use_flash_attention,
            clear_model_cache=clear_model_cache,
            only_validation_split=only_validation_split,
            few_shot=few_shot,
            num_iterations=num_iterations,
            run_with_cli=run_with_cli,
        )

        self.benchmark_config = build_benchmark_config(
            first_time=True, **self.benchmark_config_default_params.model_dump()
        )

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: dict[str, list[str]] | None = None

        # Set up the results path
        self.results_path = Path.cwd() / "scandeval_benchmark_results.jsonl"

        adjust_logging_level(verbose=self.benchmark_config.verbose)

    @property
    def benchmark_results(self) -> list[BenchmarkResult]:
        """The benchmark results."""
        if self.results_path.exists():
            with self.results_path.open() as f:
                return [
                    BenchmarkResult.from_dict(json.loads(line))
                    for line in f
                    if line.strip()
                ]
        else:
            return list()

    def benchmark(
        self,
        model: list[str] | str | None = None,
        task: str | list[str] | None = None,
        dataset: list[str] | str | None = None,
        progress_bar: bool | None = None,
        save_results: bool | None = None,
        language: str | list[str] | None = None,
        model_language: str | list[str] | None = None,
        dataset_language: str | list[str] | None = None,
        framework: Framework | str | None = None,
        device: Device | None = None,
        batch_size: int | None = None,
        evaluate_train: bool | None = None,
        raise_errors: bool | None = None,
        cache_dir: str | None = None,
        token: bool | str | None = None,
        openai_api_key: str | None = None,
        azure_openai_api_key: str | None = None,
        azure_openai_endpoint: str | None = None,
        azure_openai_api_version: str | None = None,
        force: bool | None = None,
        verbose: bool | None = None,
        trust_remote_code: bool | None = None,
        load_in_4bit: bool | None = None,
        use_flash_attention: bool | None = None,
        clear_model_cache: bool | None = None,
        only_validation_split: bool | None = None,
        few_shot: bool | None = None,
        num_iterations: int | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmarks models on datasets.

        Args:
            model:
                The full Hugging Face Hub path(s) to the pretrained transformer model.
                The specific model version to use can be added after the suffix '@':
                "model@v1.0.0". It can be a branch name, a tag name, or a commit id,
                and defaults to the latest version if not specified. If None then all
                relevant model IDs will be benchmarked. Defaults to None.
            task:
                The tasks benchmark the model(s) on. Mutually exclusive with `dataset`.
                If both `task` and `dataset` are None then all datasets will be
                benchmarked. Defaults to None.
            dataset:
                The datasets to benchmark on. Mutually exclusive with `task`. If both
                `task` and `dataset` are None then all datasets will be benchmarked.
                Defaults to None.
            progress_bar:
                Whether progress bars should be shown. Defaults to the value specified
                when initialising the benchmarker.
            save_results:
                Whether to save the benchmark results to
                'scandeval_benchmark_results.jsonl'. Defaults to the value specified
                when initialising the benchmarker.
            language:
                The language codes of the languages to include, both for models and
                datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
                to 'all' if all languages (also non-Scandinavian) should be considered.
                Defaults to the value specified when initialising the benchmarker.
            model_language:
                The language codes of the languages to include for models. If specified
                then this overrides the `language` parameter for model languages.
                Defaults to the value specified when initialising the benchmarker.
            dataset_language:
                The language codes of the languages to include for datasets. If
                specified then this overrides the `language` parameter for dataset
                languages. Defaults to the value specified when initialising the
                benchmarker.
            framework:
                The model framework to use. Only relevant if `model-id` refers to a
                local path. Otherwise, the framework will be set automatically.
                Defaults to the value specified when initialising the benchmarker.
            device:
                The device to use for benchmarking. Defaults to the value specified when
                initialising the benchmarker.
            batch_size:
                The batch size to use. Defaults to the value specified when initialising
                the benchmarker.
            evaluate_train:
                Whether to evaluate the training set as well. Defaults to the value
                specified when initialising the benchmarker.
            raise_errors:
                Whether to raise errors instead of skipping the model evaluation.
            cache_dir:
                Directory to store cached models. Defaults to the value specified when
                initialising the benchmarker.
            token:
                The authentication token for the Hugging Face Hub. If a boolean value is
                specified then the token will be fetched from the Hugging Face CLI, where
                the user has logged in through `huggingface-cli login`. If a string is
                specified then it will be used as the token. Defaults to the value
                specified when initialising the benchmarker.
            openai_api_key:
                The OpenAI API key to use for authentication. If None, then this will be
                loaded from the environment variable `OPENAI_API_KEY`. Defaults to the
                value specified when initialising the benchmarker.
            azure_openai_api_key:
                The Azure OpenAI API key to use for authentication. If None, then this
                will be loaded from the environment variable `AZURE_OPENAI_API_KEY`.
                Defaults to the value specified when initialising the benchmarker.
            azure_openai_endpoint:
                The Azure OpenAI endpoint to use for authentication. If None, then this
                will be loaded from the environment variable `AZURE_OPENAI_ENDPOINT`.
                Defaults to the value specified when initialising the benchmarker.
            azure_openai_api_version:
                The api version for the Azure OpenAI API, e.g. "2023-12-01-preview". If
                None then the environment varaible `AZURE_OPENAI_API_VERSION` will be used.
                Defaults to the value specified when initialising the benchmarker.
            force:
                Whether to force evaluations of models, even if they have been
                benchmarked already. Defaults to the value specified when initialising
                the benchmarker.
            verbose:
                Whether to output additional output. Defaults to the value specified when
                initialising the benchmarker.
            trust_remote_code:
                Whether to trust remote code when loading models. Defaults to the value
                specified when initialising the benchmarker.
            load_in_4bit:
                Whether to load models in 4-bit precision. If None then this will be done
                if CUDA is available and the model is a decoder model. Defaults to the
                value specified when initialising the benchmarker.
            use_flash_attention:
                Whether to use Flash Attention. Defaults to the value specified when
                initialising the benchmarker.
            clear_model_cache:
                Whether to clear the model cache after benchmarking each model. Defaults
                to the value specified when initialising the benchmarker.
            only_validation_split:
                Whether to only evaluate the validation split of the datasets. Defaults
                to the value specified when initialising the benchmarker.
            few_shot:
                Whether to only evaluate the model using few-shot evaluation. Only
                relevant if the model is generative. Defaults to the value specified
                when initialising the benchmarker.
            num_iterations:
                The number of times each model should be evaluated. This is only meant
                to be used for power users, and scores will not be allowed on the
                leaderboards if this is changed. Defaults to the value specified when
                initialising the benchmarker.

        Returns:
            A list of benchmark results.

        Raises:
            ValueError:
                If both `task` and `dataset` are specified.
        """
        if task is not None and dataset is not None:
            raise ValueError("Only one of `task` and `dataset` can be specified.")

        benchmark_config_params = deepcopy(self.benchmark_config_default_params)
        if task is not None:
            benchmark_config_params.task = task
            benchmark_config_params.dataset = None
        if dataset is not None:
            benchmark_config_params.dataset = dataset
            benchmark_config_params.task = None
        if progress_bar is not None:
            benchmark_config_params.progress_bar = progress_bar
        if save_results is not None:
            benchmark_config_params.save_results = save_results
        if language is not None:
            benchmark_config_params.language = language
        if model_language is not None:
            benchmark_config_params.model_language = model_language
        if dataset_language is not None:
            benchmark_config_params.dataset_language = dataset_language
        if framework is not None:
            benchmark_config_params.framework = framework
        if device is not None:
            benchmark_config_params.device = device
        if batch_size is not None:
            benchmark_config_params.batch_size = batch_size
        if evaluate_train is not None:
            benchmark_config_params.evaluate_train = evaluate_train
        if raise_errors is not None:
            benchmark_config_params.raise_errors = raise_errors
        if cache_dir is not None:
            benchmark_config_params.cache_dir = cache_dir
        if token is not None:
            benchmark_config_params.token = token
        if openai_api_key is not None:
            benchmark_config_params.openai_api_key = openai_api_key
        if azure_openai_api_key is not None:
            benchmark_config_params.azure_openai_api_key = azure_openai_api_key
        if azure_openai_endpoint is not None:
            benchmark_config_params.azure_openai_endpoint = azure_openai_endpoint
        if azure_openai_api_version is not None:
            benchmark_config_params.azure_openai_api_version = azure_openai_api_version
        if force is not None:
            benchmark_config_params.force = force
        if verbose is not None:
            benchmark_config_params.verbose = verbose
        if trust_remote_code is not None:
            benchmark_config_params.trust_remote_code = trust_remote_code
        if load_in_4bit is not None:
            benchmark_config_params.load_in_4bit = load_in_4bit
        if use_flash_attention is not None:
            benchmark_config_params.use_flash_attention = use_flash_attention
        if clear_model_cache is not None:
            benchmark_config_params.clear_model_cache = clear_model_cache
        if only_validation_split is not None:
            benchmark_config_params.only_validation_split = only_validation_split
        if few_shot is not None:
            benchmark_config_params.few_shot = few_shot
        if num_iterations is not None:
            benchmark_config_params.num_iterations = num_iterations

        benchmark_config = build_benchmark_config(
            **benchmark_config_params.model_dump()
        )
        adjust_logging_level(verbose=benchmark_config.verbose)

        if benchmark_config.clear_model_cache:
            clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

        model_ids = self._prepare_model_ids(
            model=model,
            model_languages=benchmark_config.model_languages,
            token=benchmark_config.token,
        )
        dataset_configs = prepare_dataset_configs(
            dataset_names=benchmark_config.datasets
        )

        current_benchmark_results: list[BenchmarkResult] = list()
        for m_id in model_ids:
            m_id = m_id.rstrip(" /")
            loaded_model = None
            loaded_tokenizer = None

            for dataset_config in dataset_configs:
                # Skip if we have already benchmarked this model on this dataset and
                # we are not forcing the benchmark
                if not benchmark_config.force and model_has_been_benchmarked(
                    model_id=m_id,
                    dataset=dataset_config.name,
                    few_shot=benchmark_config.few_shot,
                    validation_split=benchmark_config.only_validation_split,
                    benchmark_results=self.benchmark_results,
                ):
                    logger.debug(
                        f"Skipping benchmarking {m_id} on {dataset_config.pretty_name},"
                        " as it has already been benchmarked."
                    )
                    continue

                # Benchmark a single model on a single dataset
                try:
                    benchmark_output = self._benchmark_single(
                        dataset_config=dataset_config,
                        model_id=m_id,
                        raise_errors=benchmark_config.raise_errors,
                        model=loaded_model,
                        tokenizer=loaded_tokenizer,
                        benchmark_config=benchmark_config,
                    )
                except InvalidModel as e:
                    if benchmark_config.raise_errors:
                        raise e
                    logger.info(e)
                    break

                # If the benchmark was unsuccessful then skip
                if isinstance(benchmark_output, dict) and "error" in benchmark_output:
                    error_msg = benchmark_output["error"]
                    logger.info(
                        f"{m_id} could not be benchmarked on "
                        f"{dataset_config.pretty_name}. Skipping. The error message "
                        f"raised was {error_msg!r}."
                    )
                    continue

                assert isinstance(benchmark_output, tuple)
                record, loaded_model, loaded_tokenizer = benchmark_output

                # Save the benchmark results
                assert isinstance(record, BenchmarkResult)
                current_benchmark_results.append(record)
                if benchmark_config.save_results:
                    record.append_to_results(results_path=self.results_path)

            if benchmark_config.clear_model_cache:
                clear_model_cache_fn(cache_dir=benchmark_config.cache_dir)

        return current_benchmark_results

    def _prepare_model_ids(
        self,
        model: list[str] | str | None,
        model_languages: list["Language"],
        token: bool | str | None,
    ) -> list[str]:
        """Prepare the model ID(s) to be benchmarked.

        Args:
            model:
                The model ID(s) of the models to benchmark. If None then all model IDs
                will be retrieved.
            model_languages:
                The languages of the models to fetch.
            token:
                The authentication token for the Hugging Face Hub.

        Returns:
            The prepared list of model IDs.
        """
        model_ids: list[str]

        # If `model_id` is not specified, then fetch all the relevant model IDs
        if model is None:
            model_ids = self._get_model_ids(languages=model_languages, token=token)

        # Otherwise, if `model_id` is a string, ensure that it is a list
        elif isinstance(model, str):
            model_ids = [model]

        # Otherwise `model_id` is already a list, so we do nothing
        else:
            model_ids = model

        # Reorder the `model_ids` list to include the ones present in the benchmark
        # results first
        benchmarked_model_ids = [
            re.sub(r"\(.+\)", "", record.model).strip()
            for record in self.benchmark_results
        ]
        model_ids_sorted = [m_id for m_id in model_ids if m_id in benchmarked_model_ids]
        model_ids_sorted += [
            m_id for m_id in model_ids if m_id not in benchmarked_model_ids
        ]

        return model_ids_sorted

    def _benchmark_single(
        self,
        dataset_config: "DatasetConfig",
        model_id: str,
        raise_errors: bool,
        model: "GenerativeModel | None",
        tokenizer: "Tokenizer | None",
        benchmark_config: BenchmarkConfig,
    ) -> (
        tuple[BenchmarkResult, "GenerativeModel | None", "Tokenizer | None"]
        | dict[str, str]
    ):
        """Benchmark a single model on a single dataset.

        Args:
            dataset_config:
                The dataset configuration to use.
            model_id:
                The model ID to use.
            raise_errors:
                Whether to raise errors instead of skipping the model evaluation.
            model:
                The pre-loaded model, if available.
            tokenizer:
                The pre-loaded tokenizer, if available.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The benchmark result, or a dictionary containing an error message.
        """
        logger.info(f"Benchmarking {model_id} on {dataset_config.pretty_name}")
        if dataset_config.unofficial:
            logger.info(
                f"Note that the {dataset_config.name!r} dataset is unofficial, "
                "meaning that the resulting evaluation will not be included in the "
                "official leaderboard."
            )
        while True:
            try:
                dataset_factory = DatasetFactory(benchmark_config=benchmark_config)
                dataset = dataset_factory.build_dataset(dataset_config)
                results, metadata_dict, model, tokenizer = dataset(
                    model_id=model_id, model=model, tokenizer=tokenizer
                )
                record = BenchmarkResult(
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
                return record, model, tokenizer

            except InvalidBenchmark as e:
                # If the model ID is not valid then raise an error, if specified
                model_err_msg = "does not exist on the Hugging Face Hub"
                if raise_errors and model_err_msg in str(e):
                    raise e

                # Otherwise, if the error is due to Hugging Face Hub being down, then
                # wait a bit and try again
                if "The Hugging Face Hub seems to be down." in str(e):
                    wait_time = 30
                    logger.debug(
                        "The Hugging Face Hub seems to be down. Retrying in "
                        f"{wait_time} seconds."
                    )
                    sleep(wait_time)
                    continue

                # Otherwise, if the error is due to the MPS fallback not being enabled,
                # then raise an error asking the user to enable it
                elif "PYTORCH_ENABLE_MPS_FALLBACK" in str(e):
                    raise RuntimeError(
                        "The benchmark failed because the environment variable "
                        "`PYTORCH_ENABLE_MPS_FALLBACK` is not set. Please set this "
                        "environment variable to `1` and try again."
                    )

                # Otherwise, raise the error or return the error message
                else:
                    if raise_errors:
                        raise e
                    return dict(error=str(e))

    def __call__(self, *args, **kwargs) -> list[BenchmarkResult]:
        """Call the benchmarker. See `Benchmarker.benchmark`."""
        return self.benchmark(*args, **kwargs)

    def _get_model_ids(
        self, languages: list["Language"], token: bool | str | None
    ) -> list[str]:
        """Get list of model IDs from the Hugging Face Hub.

        Args:
            languages:
                The languages of the models to fetch.
            token:
                The authentication token for the Hugging Face Hub.

        Returns:
            List of model IDs.
        """
        # Specify boolean variables determining whether the input variables are new
        new_languages = self._model_lists is not None and any(
            lang.code not in self._model_lists for lang in languages
        )

        # If the model lists have not been fetched already, then do it
        if self._model_lists is None or new_languages:
            self._model_lists = get_huggingface_model_lists(
                languages=languages, token=token
            )

        # Extract all the model IDs from the model lists, for the chosen languages
        model_ids: list[str] = list()
        for language in languages:
            model_ids.extend(self._model_lists[language.code])

        # Add the multilingual models
        model_ids.extend(self._model_lists["multilingual"])

        # Add the fresh models
        model_ids.extend(self._model_lists["fresh"])

        # Remove duplicate model IDs
        model_ids = list(set(model_ids))

        return model_ids


def model_has_been_benchmarked(
    model_id: str,
    dataset: str,
    few_shot: bool,
    validation_split: bool,
    benchmark_results: list[BenchmarkResult],
) -> bool:
    """Checks whether a model has already been benchmarked on a dataset.

    Args:
        model_id:
            The model ID.
        dataset:
            The dataset.
        few_shot:
            Whether the model was evaluated using few-shot evaluation.
        validation_split:
            Whether the model was evaluated on the validation split.
        benchmark_results:
            The benchmark results.

    Returns:
        Whether the model has already been evaluated on the dataset.
    """
    for record in benchmark_results:
        same_evaluation = record.model == model_id and record.dataset == dataset
        same_validation_split_setting = record.validation_split == validation_split
        same_few_shot_setting = record.few_shot == few_shot or not record.generative
        if same_evaluation and same_validation_split_setting and same_few_shot_setting:
            return True
    return False


def adjust_logging_level(verbose: bool) -> None:
    """Adjust the logging level based on verbosity.

    Args:
        verbose:
            Whether to output additional output.
    """
    if hasattr(sys, "_called_from_test"):
        logging_level = logging.CRITICAL
    elif verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logger.setLevel(logging_level)


def clear_model_cache_fn(cache_dir: str) -> None:
    """Clear the model cache.

    Note that this will not remove the stored completions.

    Args:
        cache_dir:
            The path to the cache directory.
    """
    model_cache_path = Path(cache_dir) / "model_cache"
    model_cache_path.mkdir(parents=True, exist_ok=True)
    for model_dir in model_cache_path.iterdir():
        if model_dir.is_dir():
            for sub_model_dir in model_dir.iterdir():
                if sub_model_dir.is_dir():
                    rmtree(sub_model_dir)


def prepare_dataset_configs(dataset_names: list[str]) -> list["DatasetConfig"]:
    """Prepare the dataset configuration(s) to be benchmarked.

    Args:
        dataset_names:
            The dataset names to benchmark.

    Returns:
        The prepared list of model IDs.
    """
    return [
        cfg for cfg in get_all_dataset_configs().values() if cfg.name in dataset_names
    ]
