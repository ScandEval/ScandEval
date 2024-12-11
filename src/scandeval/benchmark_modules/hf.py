"""Encoder models from the Hugging Face Hub."""

import collections.abc as c
import logging
import typing as t
from functools import cached_property, partial
from json import JSONDecodeError
from time import sleep

import torch
from datasets import DatasetDict
from huggingface_hub import HfApi
from huggingface_hub import whoami as hf_whoami
from huggingface_hub.hf_api import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.utils import (
    GatedRepoError,
    HFValidationError,
    LocalTokenNotFoundError,
)
from requests.exceptions import RequestException
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)
from urllib3.exceptions import RequestError

from ..constants import DUMMY_FILL_VALUE, GENERATIVE_MODEL_TASKS, GENERATIVE_TAGS
from ..data_models import BenchmarkConfig, DatasetConfig, HFModelInfo, ModelConfig, Task
from ..enums import BatchingPreference, Framework, ModelType
from ..exceptions import (
    HuggingFaceHubDown,
    InvalidBenchmark,
    InvalidModel,
    NeedsAdditionalArgument,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
    NoInternetConnection,
)
from ..languages import get_all_languages
from ..task_utils import question_answering, token_classification
from ..types import ExtractLabelsFunction
from ..utils import (
    block_terminal_output,
    create_model_cache_dir,
    get_class_by_name,
    internet_connection_available,
)
from .base import BenchmarkModule

logger = logging.getLogger("scandeval")


class HuggingFaceEncoderModel(BenchmarkModule):
    """An encoder model from the Hugging Face Hub."""

    _is_generative = False
    batching_preference = BatchingPreference.NO_PREFERENCE

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        model, tokenizer = self._load_model_and_tokenizer()
        self._model: PreTrainedModel = model
        self._tokenizer: PreTrainedTokenizer = tokenizer
        self._log_metadata()

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        hf_api = HfApi()
        try:
            repo_info = hf_api.model_info(
                repo_id=self.model_config.adapter_base_model_id
                or self.model_config.model_id,
                revision=self.model_config.revision,
                token=self.benchmark_config.api_key or True,
            )
        except (
            RepositoryNotFoundError,
            RevisionNotFoundError,
            RequestException,
            HFValidationError,
        ):
            repo_info = None

        if (
            repo_info is not None
            and hasattr(repo_info, "safetensors")
            and repo_info.safetensors is not None
            and "total" in repo_info.safetensors
        ):
            num_params = repo_info.safetensors["total"]
        elif (
            hasattr(self._model.config, "num_params")
            and self._model.config.num_params is not None
        ):
            num_params = self._model.config.num_params
        else:
            num_params = sum(p.numel() for p in self._model.parameters())
        return num_params

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        if (
            hasattr(self._model.config, "vocab_size")
            and self._model.config.vocab_size is not None
        ):
            vocab_size = self._model.config.vocab_size
        elif (
            hasattr(self._tokenizer, "vocab_size")
            and self._tokenizer.vocab_size is not None
        ):
            vocab_size = self._tokenizer.vocab_size
        else:
            vocab_size = -1
        return vocab_size

    @cached_property
    def model_max_length(self) -> int:
        """The maximum context length of the model.

        Returns:
            The maximum context length of the model.
        """
        all_max_lengths: list[int] = list()

        # Add the registered max length of the tokenizer
        if hasattr(
            self._tokenizer, "model_max_length"
        ) and self._tokenizer.model_max_length < int(1e30):
            all_max_lengths.append(self._tokenizer.model_max_length)

        # Add the max length derived from the model's input sizes
        if hasattr(self._tokenizer, "max_model_input_sizes"):
            all_max_lengths.extend(
                [
                    size
                    for size in self._tokenizer.max_model_input_sizes.values()
                    if size is not None
                ]
            )

        # Add max length candidates from the model's configuration
        candidate_config_max_lengths = [
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
            "sliding_window",
            "sliding_window_size",
            "n_positions",
        ]
        for candidate_config_max_length in candidate_config_max_lengths:
            if (
                hasattr(self._model.config, candidate_config_max_length)
                and (value := getattr(self._model.config, candidate_config_max_length))
                is not None
            ):
                all_max_lengths.append(value)

        # To avoid models having artificially low max lengths, we remove any max lengths
        # that are less than 128
        all_max_lengths = [
            max_length for max_length in all_max_lengths if max_length >= 128
        ]

        if len(list(all_max_lengths)) > 0:
            model_max_length = min(list(all_max_lengths))
        else:
            model_max_length = -1

        return model_max_length

    @cached_property
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        match self.dataset_config.task.supertask:
            case "sequence-classification" | "text-to-text" | "question-answering":
                return DataCollatorWithPadding(self._tokenizer, padding="longest")
            case "token-classification":
                return DataCollatorForTokenClassification(
                    tokenizer=self._tokenizer, label_pad_token_id=-100
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {self.dataset_config.task.supertask}."
                )

    @cached_property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        raise NotImplementedError(
            "The `extract_labels_from_generation` property has not been implemented "
            "for Hugging Face Encoder models."
        )

    @cached_property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        match self.dataset_config.task.supertask:
            case "sequence-classification" | "text-to-text" | "token-classification":
                return Trainer
            case "question-answering":
                return question_answering.QuestionAnsweringTrainer
            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {self.dataset_config.task.supertask}."
                )

    def prepare_dataset(
        self, dataset: DatasetDict, task: Task, itr_idx: int
    ) -> DatasetDict:
        """Prepare the dataset for the model.

        This includes things like tokenisation.

        Args:
            dataset:
                The dataset to prepare.
            task:
                The task to prepare the dataset for.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The prepared dataset.
        """

        def numericalise_labels(examples: dict):
            if "label" in examples:
                try:
                    examples["label"] = [
                        self._model.config.label2id[lbl.lower()]
                        for lbl in examples["label"]
                    ]
                except KeyError:
                    raise InvalidBenchmark(
                        f"One of the labels in the dataset, "
                        f"{examples['label'].lower()}, does not occur in the "
                        f"label2id dictionary {self._model.config.label2id}."
                    )
            return examples

        def tokenise(examples: dict):
            return self._tokenizer(text=examples["text"], truncation=True, padding=True)

        match task.supertask:
            case "sequence-classification":
                dataset = dataset.map(
                    numericalise_labels, batched=True, load_from_cache_file=False
                ).map(tokenise, batched=True, load_from_cache_file=False)

            case "text-to-text":
                dataset = dataset.map(
                    tokenise,
                    batched=True,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )

            case "token-classification":
                dataset = dataset.map(
                    partial(
                        token_classification.tokenize_and_align_labels,
                        tokenizer=self._tokenizer,
                        label2id=self._model.config.label2id,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )

            case "question-answering":
                dataset = DatasetDict(
                    dict(
                        train=dataset["train"].map(
                            partial(
                                question_answering.prepare_train_examples,
                                tokenizer=self._tokenizer,
                            ),
                            batched=True,
                            batch_size=10,
                            remove_columns=dataset["test"].column_names,
                            load_from_cache_file=False,
                            keep_in_memory=True,
                        ),
                        val=dataset["val"].map(
                            partial(
                                question_answering.prepare_train_examples,
                                tokenizer=self._tokenizer,
                            ),
                            batched=True,
                            batch_size=10,
                            remove_columns=dataset["test"].column_names,
                            load_from_cache_file=False,
                            keep_in_memory=True,
                        ),
                        test=dataset["test"].map(
                            partial(
                                question_answering.prepare_test_examples,
                                tokenizer=self._tokenizer,
                            ),
                            batched=True,
                            batch_size=10,
                            remove_columns=dataset["test"].column_names,
                            load_from_cache_file=False,
                            keep_in_memory=True,
                        ),
                    )
                )

                # The Trainer hides the columns that are not used by the model (here
                # `id` and `offset_mapping` which we will need for our post-processing),
                # so we put them back
                for split_name, split in dataset.items():
                    dataset[split_name].set_format(
                        type=split.format["type"], columns=list(split.features.keys())
                    )

            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {task.supertask}."
                )

        return dataset

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        model_id, revision = (
            model_id.split("@") if "@" in model_id else (model_id, "main")
        )
        model_info = get_model_repo_info(
            model_id=model_id, revision=revision, benchmark_config=benchmark_config
        )
        return (
            model_info is not None
            and model_info.pipeline_tag not in GENERATIVE_MODEL_TASKS
        )

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        model_id, revision = (
            model_id.split("@") if "@" in model_id else (model_id, "main")
        )
        model_info = get_model_repo_info(
            model_id=model_id, revision=revision, benchmark_config=benchmark_config
        )
        if model_info is None:
            raise InvalidModel(f"The model {model_id!r} could not be found.")

        framework = Framework.PYTORCH
        if "pytorch" in model_info.tags:
            pass
        elif "jax" in model_info.tags:
            framework = Framework.JAX
        elif "spacy" in model_info.tags:
            raise InvalidModel("SpaCy models are not supported.")
        elif any(tag in model_info.tags for tag in {"tf", "tensorflow", "keras"}):
            raise InvalidModel("TensorFlow/Keras models are not supported.")

        language_mapping = get_all_languages()
        language_codes = list(language_mapping.keys())

        model_config = ModelConfig(
            model_id=model_id,
            revision=revision,
            framework=framework,
            task=model_info.pipeline_tag,
            languages=[
                language_mapping[tag]
                for tag in model_info.tags
                if tag in language_codes
            ],
            model_type=ModelType.HF_HUB_ENCODER,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=None,
        )

        return model_config

    def _load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer.

        Returns:
            The loaded model and tokenizer.
        """
        config: "PretrainedConfig"
        block_terminal_output()

        model_id = self.model_config.model_id
        supertask = self.dataset_config.task.supertask
        from_flax = self.model_config.framework == Framework.JAX
        ignore_mismatched_sizes = False

        config = load_hf_model_config(
            model_id=model_id,
            num_labels=self.dataset_config.num_labels,
            id2label=self.dataset_config.id2label,
            label2id=self.dataset_config.label2id,
            revision=self.model_config.revision,
            model_cache_dir=self.model_config.model_cache_dir,
            api_key=self.benchmark_config.api_key,
            trust_remote_code=self.benchmark_config.trust_remote_code,
            run_with_cli=self.benchmark_config.run_with_cli,
        )

        model_kwargs = dict(
            config=config,
            from_flax=from_flax,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            revision=self.model_config.revision,
            token=self.benchmark_config.api_key or True,
            cache_dir=self.model_config.model_cache_dir,
            trust_remote_code=self.benchmark_config.trust_remote_code,
            torch_dtype=get_torch_dtype(
                device=self.benchmark_config.device,
                torch_dtype_is_set=config.to_dict().get("torch_dtype") is not None,
                bf16_available=(
                    torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                ),
            ),
        )

        # These are used when a timeout occurs
        attempts_left = 5

        model: PreTrainedModel | None = None
        while True:
            try:
                # Get the model class associated with the supertask
                model_cls_or_none: t.Type["PreTrainedModel"] | None = get_class_by_name(
                    class_name=f"auto-model-for-{supertask}", module_name="transformers"
                )

                # If the model class could not be found then raise an error
                if not model_cls_or_none:
                    raise InvalidBenchmark(
                        f"The supertask {supertask!r} does not correspond to a "
                        "Hugging Face AutoModel type (such as "
                        "`AutoModelForSequenceClassification`)."
                    )

                # If the model is a DeBERTaV2 model then we ensure that
                # `pooler_hidden_size` is the same size as `hidden_size`
                if config.model_type == "deberta-v2":
                    config.pooler_hidden_size = config.hidden_size

                try:
                    model_or_tuple = model_cls_or_none.from_pretrained(
                        self.model_config.model_id, **model_kwargs
                    )
                except (KeyError, RuntimeError) as e:
                    if not model_kwargs["ignore_mismatched_sizes"]:
                        logger.debug(
                            f"{type(e).__name__} occurred during the loading "
                            f"of the {model_id!r} model. Retrying with "
                            "`ignore_mismatched_sizes` set to True."
                        )
                        model_kwargs["ignore_mismatched_sizes"] = True
                        continue
                    else:
                        raise InvalidModel(str(e))
                except (TimeoutError, RequestError):
                    attempts_left -= 1
                    if attempts_left == 0:
                        raise InvalidModel(
                            "The model could not be loaded after 5 attempts."
                        )
                    logger.info(f"Couldn't load the model {model_id!r}. Retrying.")
                    sleep(5)
                    continue

                if isinstance(model_or_tuple, tuple):
                    model = model_or_tuple[0]
                else:
                    model = model_or_tuple
                break

            except (OSError, ValueError) as e:
                # If `from_flax` is False but only Flax models are available then
                # try again with `from_flax` set to True
                if not from_flax and "Use `from_flax=True` to load this model" in str(
                    e
                ):
                    from_flax = True
                    continue

                if "checkpoint seems to be incorrect" in str(e):
                    raise InvalidModel(
                        f"The model {model_id!r} has an incorrect checkpoint."
                    )
                if "trust_remote_code" in str(e):
                    raise InvalidModel(
                        f"Loading the model {model_id!r} needs to trust remote code. "
                        "If you trust the suppliers of this model, then you can enable "
                        "this by setting the `--trust-remote-code` flag."
                    )
                raise InvalidModel(
                    f"The model {model_id!r} could not be loaded. The error was {e!r}."
                )

        assert model is not None

        model.eval()
        model.to(self.benchmark_config.device)

        if isinstance(model, PreTrainedModel) and supertask == "question-answering":
            model = setup_model_for_question_answering(model=model)

        tokenizer = load_tokenizer(
            model=model,
            model_id=model_id,
            trust_remote_code=self.benchmark_config.trust_remote_code,
        )

        model, tokenizer = align_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            raise_errors=self.benchmark_config.raise_errors,
        )

        return model, tokenizer


def get_model_repo_info(
    model_id: str, revision: str, benchmark_config: BenchmarkConfig
) -> HFModelInfo | None:
    """Get the information about the model from the Hugging Face Hub.

    Args:
        model_id:
            The model ID.
        revision:
            The revision of the model.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The information about the model, or None if the model could not be found.
    """
    hf_api = HfApi(token=benchmark_config.api_key or True)
    model_id, revision = model_id.split("@") if "@" in model_id else (model_id, "main")

    try:
        model_info = hf_api.model_info(repo_id=model_id, revision=revision)

    # Case where the model is gated; note this to the user
    except (GatedRepoError, LocalTokenNotFoundError) as e:
        try:
            hf_whoami()
            logger.warning(
                f"Could not access the model {model_id} with the revision "
                f"{revision}. The error was {str(e)!r}."
            )
            return None
        except LocalTokenNotFoundError:
            raise NeedsAdditionalArgument(
                cli_argument="--api-key",
                script_argument="api_key=<your-api-key>",
                run_with_cli=benchmark_config.run_with_cli,
            )

    # Case where the model could not be found
    except (RepositoryNotFoundError, HFValidationError):
        return None

    # Other internet-related errors
    except (OSError, RequestException):
        if internet_connection_available():
            raise HuggingFaceHubDown()
        else:
            raise NoInternetConnection()

    tags = model_info.tags or list()

    has_base_model_tag = any(
        tag.startswith("base_model:") and tag.count(":") == 1 for tag in tags
    )
    base_model_id: str | None = None
    if has_base_model_tag:
        has_adapter_config = model_info.siblings is not None and any(
            sibling.rfilename == "adapter_config.json"
            for sibling in model_info.siblings
        )
        if has_adapter_config:
            base_model_id = [
                tag.split(":")[1]
                for tag in tags
                if tag.startswith("base_model:") and tag.count(":") == 1
            ][0]
            base_model_info = hf_api.model_info(
                repo_id=base_model_id,
                revision=revision,
                token=benchmark_config.api_key or True,
            )
            tags += base_model_info.tags or list()
            tags = list(set(tags))

    pipeline_tag = model_info.pipeline_tag
    if pipeline_tag is None:
        if any(tag in GENERATIVE_TAGS for tag in tags):
            pipeline_tag = "text-generation"
        else:
            pipeline_tag = "fill-mask"

    return HFModelInfo(
        pipeline_tag=pipeline_tag, tags=tags, adapter_base_model_id=base_model_id
    )


def load_tokenizer(
    model: "PreTrainedModel | None", model_id: str, trust_remote_code: bool
) -> "PreTrainedTokenizer":
    """Load the tokenizer.

    Args:
        model:
            The model, which is used to determine whether to add a prefix space to
            the tokens. Can be None.
        model_id:
            The model identifier. Used for logging.
        trust_remote_code:
            Whether to trust remote code.

    Returns:
        The loaded tokenizer.
    """
    loading_kwargs: dict[str, bool | str] = dict(
        use_fast=True,
        verbose=False,
        trust_remote_code=trust_remote_code,
        padding_side="right",
        truncation_side="right",
    )

    # If the model is a subclass of a certain model types then we have to add a prefix
    # space to the tokens, by the way the model is constructed.
    if model is not None:
        prefix_models = ["Roberta", "GPT", "Deberta"]
        add_prefix = any(
            model_type in type(model).__name__ for model_type in prefix_models
        )
        if add_prefix:
            loading_kwargs["add_prefix_space"] = True

    while True:
        try:
            return AutoTokenizer.from_pretrained(model_id, **loading_kwargs)
        except (JSONDecodeError, OSError, TypeError):
            raise InvalidModel(f"Could not load tokenizer for model {model_id!r}.")
        except (TimeoutError, RequestError):
            logger.info(f"Couldn't load tokenizer for {model_id!r}. Retrying.")
            sleep(5)
            continue


def get_torch_dtype(
    device: torch.device, torch_dtype_is_set: bool, bf16_available: bool
) -> str | torch.dtype:
    """Get the torch dtype, used for loading the model.

    Args:
        device:
            The device to use.
        torch_dtype_is_set:
            Whether the torch data type is set in the model configuration.
        bf16_available:
            Whether bfloat16 is available.

    Returns:
        The torch dtype.
    """
    using_cuda = device == torch.device("cuda")
    if using_cuda and torch_dtype_is_set:
        return "auto"
    elif using_cuda and bf16_available:
        return torch.bfloat16
    elif using_cuda:
        return torch.float16
    return torch.float32


def load_hf_model_config(
    model_id: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    revision: str,
    model_cache_dir: str | None,
    api_key: str | None,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> "PretrainedConfig":
    """Load the Hugging Face model configuration.

    Args:
        model_id:
            The Hugging Face model ID.
        num_labels:
            The number of labels in the dataset.
        id2label:
            The mapping from label IDs to labels.
        label2id:
            The mapping from labels to label IDs.
        revision:
            The revision of the model.
        model_cache_dir:
            The directory to cache the model in.
        api_key:
            The Hugging Face API key.
        trust_remote_code:
            Whether to trust remote code.
        run_with_cli:
            Whether the script is being run with the CLI.

    Returns:
        The Hugging Face model configuration.
    """
    while True:
        try:
            config = AutoConfig.from_pretrained(
                model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                revision=revision,
                token=api_key or True,
                trust_remote_code=trust_remote_code,
                cache_dir=model_cache_dir,
            )
            if config.eos_token_id is not None and config.pad_token_id is None:
                if isinstance(config.eos_token_id, list):
                    config.pad_token_id = config.eos_token_id[0]
                else:
                    config.pad_token_id = config.eos_token_id
            return config
        except KeyError as e:
            key = e.args[0]
            raise InvalidModel(
                f"The model config for the model {model_id!r} could not be "
                f"loaded, as the key {key!r} was not found in the config."
            )
        except OSError as e:
            # TEMP: When the model is gated then we cannot set cache dir, for some
            # reason (transformers==4.38.2). This should be included back in when
            # this is fixed.
            if "gated repo" in str(e):
                model_cache_dir = None
                continue
            raise InvalidModel(
                f"Couldn't load model config for {model_id!r}. The error was "
                f"{e!r}. Skipping"
            )
        except (TimeoutError, RequestError):
            logger.info(f"Couldn't load model config for {model_id!r}. Retrying.")
            sleep(5)
            continue
        except ValueError as e:
            requires_trust_remote_code = "trust_remote_code" in str(e)
            if requires_trust_remote_code:
                raise NeedsAdditionalArgument(
                    cli_argument="--trust-remote-code",
                    script_argument="trust_remote_code=True",
                    run_with_cli=run_with_cli,
                )
            raise e


def setup_model_for_question_answering(model: "PreTrainedModel") -> "PreTrainedModel":
    """Setup a model for question answering.

    Args:
        model:
            The model to setup.

    Returns:
        The setup model.
    """
    # Get the models' token type embedding children, if they exist
    children = get_children_of_module(name="model", module=model)

    # If the model has token type embeddings then get them
    if children:
        # Get the list of attributes that are token type embeddings
        attribute_list = list()
        done = False
        while not done:
            for key, value in children.items():
                attribute_list.append(key)
                if isinstance(value, dict):
                    children = value
                else:
                    done = True
                break

        # Get the token type embeddings
        token_type_embeddings = model
        for attribute in attribute_list:
            token_type_embeddings = getattr(token_type_embeddings, attribute)

        # If the token type embeddings has shape (1, ...) then set the shape to
        # (2, ...) by randomly initializing the second token type embedding
        if token_type_embeddings.weight.data.shape[0] == 1:
            token_type_embeddings.weight.data = torch.cat(
                (
                    token_type_embeddings.weight.data,
                    torch.rand_like(token_type_embeddings.weight.data),
                ),
                dim=0,
            )
            token_type_embeddings.num_embeddings = 2

        # Set the model config to use the new type vocab size
        model.config.type_vocab_size = 2

    return model


def get_children_of_module(
    name: str, module: nn.Module
) -> nn.Module | dict[str, t.Any] | None:
    """Get the children of a module.

    Args:
        name:
            The name of the module.
        module:
            The module to get the children of.

    Returns:
        The children of the module, or None if the module has no children.
    """
    if len(list(module.children())) == 0:
        if name == "token_type_embeddings":
            return module
        else:
            return None
    else:
        submodules = dict()
        for subname, submodule in module.named_children():
            children = get_children_of_module(name=subname, module=submodule)
            if children:
                submodules[subname] = children
        return submodules


def align_model_and_tokenizer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    raise_errors: bool = False,
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """Aligns the model and the tokenizer.

    Args:
        model:
            The model to fix.
        tokenizer:
            The tokenizer to fix.
        raise_errors:
            Whether to raise errors instead of trying to fix them silently.

    Returns:
        The fixed model and tokenizer.
    """
    model_max_length = get_model_max_length(model=model, tokenizer=tokenizer)

    # Ensure that the model max length is at least 5,000, to avoid OOM errors
    model_max_length = min(model_max_length, 5_000)

    if model_max_length > 0:
        tokenizer.model_max_length = model_max_length
    else:
        tokenizer.model_max_length = 512

    # If we're not dealing with a generative model then we move it to CPU to avoid OOM
    # errors
    device: torch.device = torch.device("cpu")
    model_device = model.device
    model.to(device)

    # Manually check that this model max length is valid for the model, and adjust
    # otherwise
    initial_max_length = tokenizer.model_max_length
    for max_length in range(initial_max_length, 0, -1):
        tokenizer.model_max_length = max_length
        dummy_inputs = torch.full(
            size=(1, max_length),
            fill_value=DUMMY_FILL_VALUE,
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            try:
                model(dummy_inputs)
                break

            # This happens if `max_length` is too large
            except IndexError:
                continue

    # If there is a mismatch between the vocab size according to the tokenizer and
    # the vocab size according to the model, we raise an error
    if hasattr(model.config, "vocab_size") and hasattr(tokenizer, "vocab_size"):
        if model.config.vocab_size < tokenizer.vocab_size:
            if raise_errors:
                raise InvalidModel(
                    "The vocab size of the tokenizer is larger than the vocab size of "
                    "the model. As the --raise-errors option was specified, the "
                    "embeddings of the model will not be automatically adjusted."
                )
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + 1)

    if tokenizer.bos_token is None and tokenizer.eos_token is not None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    model.to(model_device)

    return model, tokenizer


def get_model_max_length(
    model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"
) -> int:
    """Get the maximum context length of a model.

    Args:
        model:
            The model.
        tokenizer:
            The tokenizer.

    Returns:
        The maximum context length.
    """
    all_max_lengths: list[int] = list()

    if tokenizer is not None:
        # Add the registered max length of the tokenizer
        if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < int(
            1e30
        ):
            all_max_lengths.append(tokenizer.model_max_length)

        # Add the max length derived from the model's input sizes
        if hasattr(tokenizer, "max_model_input_sizes"):
            all_max_lengths.extend(
                [
                    size
                    for size in tokenizer.max_model_input_sizes.values()
                    if size is not None
                ]
            )

    # Add max length candidates from the model's configuration
    candidate_config_max_lengths = [
        "max_position_embeddings",
        "model_max_length",
        "max_sequence_length",
        "sliding_window",
        "sliding_window_size",
    ]
    for candidate_config_max_length in candidate_config_max_lengths:
        if (
            hasattr(model.config, candidate_config_max_length)
            and (value := getattr(model.config, candidate_config_max_length))
            is not None
        ):
            all_max_lengths.append(value)

    # To avoid models having artificially low max lengths, we remove any max lengths
    # that are less than 128
    all_max_lengths = [
        max_length for max_length in all_max_lengths if max_length >= 128
    ]

    if len(list(all_max_lengths)) > 0:
        model_max_length = min(list(all_max_lengths))
    else:
        model_max_length = -1

    return model_max_length
