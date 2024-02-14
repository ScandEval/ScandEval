"""Model setup for Hugging Face Hub models."""

import importlib.util
import logging
import os
import warnings
from json import JSONDecodeError
from time import sleep
from typing import Type

import torch
from huggingface_hub import HfApi, ModelFilter
from huggingface_hub.hf_api import RepositoryNotFoundError
from requests.exceptions import RequestException
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from urllib3.exceptions import RequestError

from ..config import BenchmarkConfig, DatasetConfig, ModelConfig
from ..enums import Framework, ModelType
from ..exceptions import (
    FlashAttentionNotInstalled,
    HuggingFaceHubDown,
    InvalidBenchmark,
    InvalidModel,
    NeedsExtraInstalled,
    NoInternetConnection,
)
from ..languages import get_all_languages
from ..protocols import GenerativeModel, Tokenizer
from ..utils import (
    GENERATIVE_DATASET_SUPERTASKS,
    GENERATIVE_DATASET_TASKS,
    GENERATIVE_MODEL_TASKS,
    block_terminal_output,
    create_model_cache_dir,
    get_class_by_name,
    internet_connection_available,
    model_is_generative,
)
from ..vllm_models import VLLMModel
from .utils import align_model_and_tokenizer, setup_model_for_question_answering

logger = logging.getLogger(__package__)


class HFModelSetup:
    """Model setup for Hugging Face Hub models.

    Args:
        benchmark_config:
            The benchmark configuration.

    Attributes:
        benchmark_config:
            The benchmark configuration.
    """

    def __init__(self, benchmark_config: BenchmarkConfig) -> None:
        """Initialize the model setup.

        Args:
            benchmark_config:
                The benchmark configuration.
        """
        self.benchmark_config = benchmark_config

    def model_exists(self, model_id: str) -> bool | str:
        """Check if a model ID denotes a model on the Hugging Face Hub.

        Args:
            model_id:
                The model ID.

        Returns:
            Whether the model exists on the Hugging Face Hub, or the name of an extra
            that needs to be installed to check if the model exists.
        """
        # Extract the revision from the model_id, if present
        model_id, revision = (
            model_id.split("@") if "@" in model_id else (model_id, "main")
        )

        # Connect to the Hugging Face Hub API
        hf_api = HfApi()

        # Get the model info, and return it
        try:
            if isinstance(self.benchmark_config.token, bool):
                token = None
            else:
                token = self.benchmark_config.token
            hf_api.model_info(repo_id=model_id, revision=revision, token=token)
            return True

        except RepositoryNotFoundError:
            return False

        # If fetching from the Hugging Face Hub failed in a different way then throw a
        # reasonable exception
        except RequestException:
            if internet_connection_available():
                raise HuggingFaceHubDown()
            else:
                raise NoInternetConnection()

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Fetches configuration for an OpenAI model.

        Args:
            model_id:
                The model ID of the model.

        Returns:
            The model configuration.
        """
        # Extract the revision from the model ID, if it is specified
        if "@" in model_id:
            model_id_without_revision, revision = model_id.split("@", 1)
        else:
            model_id_without_revision = model_id
            revision = "main"

        # Extract the author and model name from the model ID
        author: str | None
        if "/" in model_id_without_revision:
            author, model_name = model_id_without_revision.split("/")
        else:
            author = None
            model_name = model_id_without_revision

        # Attempt to fetch model data from the Hugging Face Hub
        try:
            api: HfApi = HfApi()

            # Fetch the model metadata
            models = api.list_models(
                filter=ModelFilter(author=author, model_name=model_name),
                token=self.benchmark_config.token,
            )

            # Filter the models to only keep the one with the specified model ID
            models = [
                model for model in models if model.modelId == model_id_without_revision
            ]

            # Check that the model exists. If it does not then raise an error
            if len(models) == 0:
                raise InvalidModel(
                    f"The model {model_id} does not exist on the Hugging Face Hub."
                )

            tags: list[str] = models[0].tags

            framework = Framework.PYTORCH
            if "pytorch" in tags:
                pass
            elif "jax" in tags:
                framework = Framework.JAX
            elif "spacy" in tags:
                raise InvalidModel("SpaCy models are not supported.")
            elif "tf" in tags or "tensorflow" in tags or "keras" in tags:
                raise InvalidModel("TensorFlow/Keras models are not supported.")

            model_task: str | None = models[0].pipeline_tag
            if model_task is None:
                model_task = "fill-mask"

            language_mapping = get_all_languages()
            language_codes = list(language_mapping.keys())

            model_config = ModelConfig(
                model_id=models[0].modelId,
                framework=framework,
                task=model_task,
                languages=[
                    language_mapping[tag] for tag in tags if tag in language_codes
                ],
                revision=revision,
                model_type=ModelType.HF,
                model_cache_dir=create_model_cache_dir(
                    cache_dir=self.benchmark_config.cache_dir, model_id=model_id
                ),
            )

        # If fetching from the Hugging Face Hub failed then throw a reasonable
        # exception
        except RequestException:
            if internet_connection_available():
                raise HuggingFaceHubDown()
            else:
                raise NoInternetConnection()

        return model_config

    def load_model(
        self, model_config: ModelConfig, dataset_config: DatasetConfig
    ) -> tuple[Tokenizer, PreTrainedModel | GenerativeModel]:
        """Load an OpenAI model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.

        Returns:
            The tokenizer and model.
        """
        config: PretrainedConfig
        block_terminal_output()

        model_id = model_config.model_id
        supertask = dataset_config.task.supertask
        from_flax = model_config.framework == Framework.JAX
        ignore_mismatched_sizes = False

        if self.benchmark_config.load_in_4bit is not None:
            load_in_4bit = self.benchmark_config.load_in_4bit
        else:
            load_in_4bit = (
                model_config.task in GENERATIVE_MODEL_TASKS
                and self.benchmark_config.device == torch.device("cuda")
            )

        if load_in_4bit and importlib.util.find_spec("bitsandbytes") is None:
            raise NeedsExtraInstalled(extra="generative")

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )
            if load_in_4bit
            else None
        )

        # We hardcode that we're using flash attention for models with "mistral" in
        # their name, as this is currently one of the only ones that support it
        use_flash_attention = (
            self.benchmark_config.use_flash_attention or "mistral" in model_id.lower()
        )

        config = self._load_hf_model_config(
            model_id=model_id,
            num_labels=dataset_config.num_labels,
            id2label=dataset_config.id2label,
            label2id=dataset_config.label2id,
            revision=model_config.revision,
            model_cache_dir=model_config.model_cache_dir,
        )

        use_vllm = (
            model_config.task in GENERATIVE_MODEL_TASKS
            and self.benchmark_config.device == torch.device("cuda")
            and os.getenv("USE_VLLM", True)
        )

        if use_vllm and importlib.util.find_spec("vllm") is None:
            raise NeedsExtraInstalled(extra="generative")

        if use_vllm:
            try:
                model = VLLMModel(
                    model_config=model_config,
                    hf_model_config=config,
                    dataset_config=dataset_config,
                    model_cache_dir=model_config.model_cache_dir,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                )
            except ValueError as e:
                # If the model is too large to fit on the GPU then we simply throw an
                # informative error message
                oom_error_message = "No available memory for the cache blocks"
                if oom_error_message in str(e):
                    raise InvalidModel("The model is too large to load on the GPU.")

                if self.benchmark_config.raise_errors:
                    raise e

                # Otherwise some other error occurred, and we log it and try to load
                # the model with Hugging Face instead
                use_vllm = False
                logger.info(
                    "Failed to benchmark with vLLM - trying with the Hugging Face "
                    f"implementation instead. The error raised was {e!r}"
                )

        if not use_vllm:
            model_kwargs = dict(
                config=config,
                from_flax=from_flax,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                revision=model_config.revision,
                token=self.benchmark_config.token,
                cache_dir=model_config.model_cache_dir,
                trust_remote_code=self.benchmark_config.trust_remote_code,
                quantization_config=bnb_config,
                torch_dtype=self._get_torch_dtype(config=config),
                attn_implementation=(
                    "flash_attention_2" if use_flash_attention else None
                ),
            )

            # These are used when a timeout occurs
            attempts_left = 5

            while True:
                try:
                    # Get the model class associated with the supertask
                    if model_config.task in ["text-generation", "conversational"]:
                        model_cls_supertask = "causal-l-m"
                    elif model_config.task == "text2text-generation":
                        model_cls_supertask = "seq-2-seq-l-m"
                    elif (
                        dataset_config.task.name in GENERATIVE_DATASET_TASKS
                        or supertask in GENERATIVE_DATASET_SUPERTASKS
                    ):
                        raise InvalidBenchmark(
                            f"The {dataset_config.task.name!r} task is not supported "
                            f"for the model {model_id!r}."
                        )
                    else:
                        model_cls_supertask = supertask
                    model_cls_or_none: Type[PreTrainedModel] | None = get_class_by_name(
                        class_name=f"auto-model-for-{model_cls_supertask}",
                        module_name="transformers",
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

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        try:
                            model_or_tuple = model_cls_or_none.from_pretrained(
                                model_config.model_id, **model_kwargs
                            )
                        except ImportError as e:
                            if "flash attention" in str(e).lower():
                                raise FlashAttentionNotInstalled()
                            else:
                                raise e
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
                            logger.info(
                                f"Couldn't load the model {model_id!r}. Retrying."
                            )
                            sleep(5)
                            continue
                        except ValueError as e:
                            if "already quantized" in str(e):
                                model_kwargs["quantization_config"] = None
                                model_or_tuple = model_cls_or_none.from_pretrained(
                                    model_config.model_id, **model_kwargs
                                )
                            else:
                                raise e

                    if isinstance(model_or_tuple, tuple):
                        model = model_or_tuple[0]
                    else:
                        model = model_or_tuple
                    break

                except (OSError, ValueError) as e:
                    # If `from_flax` is False but only Flax models are available then
                    # try again with `from_flax` set to True
                    if (
                        not from_flax
                        and "Use `from_flax=True` to load this model" in str(e)
                    ):
                        from_flax = True
                        continue

                    self._handle_loading_exception(exception=e, model_id=model_id)

        if supertask == "question-answering":
            model = setup_model_for_question_answering(model=model)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            tokenizer = self._load_tokenizer(model=model, model_id=model_id)

        if use_vllm:
            model.set_tokenizer(tokenizer=tokenizer)

        model, tokenizer = align_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            generation_length=dataset_config.max_generated_tokens,
            raise_errors=self.benchmark_config.raise_errors,
        )

        model.eval()
        if not load_in_4bit:
            model.to(self.benchmark_config.device)

        return tokenizer, model

    def _get_torch_dtype(self, config: PretrainedConfig) -> str | None:
        """Get the torch dtype, used for loading the model.

        Args:
            config (PretrainedConfig):
                The Hugging Face model configuration.

        Returns:
            The torch dtype.
        """
        using_cuda = self.benchmark_config.device == torch.device("cuda")
        torch_dtype_is_set = config.to_dict().get("torch_dtype") is not None
        torch_dtype = "auto" if using_cuda and torch_dtype_is_set else None
        return torch_dtype

    def _load_hf_model_config(
        self,
        model_id: str,
        num_labels: int,
        id2label: dict[int, str] | list[str],
        label2id: dict[str, int],
        revision: str,
        model_cache_dir: str,
    ) -> PretrainedConfig:
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

        Returns:
            The Hugging Face model configuration.
        """
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    config = AutoConfig.from_pretrained(
                        model_id,
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id=label2id,
                        revision=revision,
                        cache_dir=model_cache_dir,
                        token=self.benchmark_config.token,
                        trust_remote_code=self.benchmark_config.trust_remote_code,
                    )
                if config.eos_token_id is not None and config.pad_token_id is None:
                    config.pad_token_id = config.eos_token_id
                return config
            except KeyError as e:
                key = e.args[0]
                raise InvalidModel(
                    f"The model config for the model {model_id!r} could not be "
                    f"loaded, as the key {key!r} was not found in the config."
                )
            except OSError as e:
                raise InvalidModel(
                    f"Couldn't load model config for {model_id!r}. The error was "
                    f"{e!r}. Skipping"
                )
            except (TimeoutError, RequestError):
                logger.info(f"Couldn't load model config for {model_id!r}. Retrying.")
                sleep(5)
                continue

    def _load_tokenizer(
        self, model: PreTrainedModel | GenerativeModel, model_id: str
    ) -> Tokenizer:
        """Load the tokenizer.

        Args:
            model (PreTrainedModel or GenerativeModel):
                The model, which is used to determine whether to add a prefix space to
                the tokens.
            model_id (str):
                The model identifier. Used for logging.

        Returns:
            Tokenizer:
                The loaded tokenizer.
        """
        # If the model is a subclass of a RoBERTa model then we have to add a prefix
        # space to the tokens, by the way the model is constructed.
        prefix_models = ["Roberta", "GPT", "Deberta"]
        prefix = any(model_type in type(model).__name__ for model_type in prefix_models)
        padding_side = "left" if model_is_generative(model=model) else "right"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            while True:
                try:
                    return AutoTokenizer.from_pretrained(
                        model_id,
                        add_prefix_space=prefix,
                        use_fast=True,
                        verbose=False,
                        padding_side=padding_side,
                        truncation_side=padding_side,
                        trust_remote_code=self.benchmark_config.trust_remote_code,
                    )
                except (JSONDecodeError, OSError, TypeError):
                    raise InvalidModel(
                        f"Could not load tokenizer for model {model_id!r}."
                    )
                except (TimeoutError, RequestError):
                    logger.info(f"Couldn't load tokenizer for {model_id!r}. Retrying.")
                    sleep(5)
                    continue

    @staticmethod
    def _handle_loading_exception(exception: Exception, model_id: str) -> None:
        if "checkpoint seems to be incorrect" in str(exception):
            raise InvalidModel(f"The model {model_id!r} has an incorrect checkpoint.")
        if "trust_remote_code" in str(exception):
            raise InvalidModel(
                f"Loading the model {model_id!r} needs to trust remote code. "
                "If you trust the suppliers of this model, then you can enable "
                "this by setting the `--trust-remote-code` flag."
            )
        raise InvalidModel(
            f"The model {model_id} either does not exist on the Hugging Face "
            "Hub, or it has no frameworks registered, or it is a private "
            "model. If it *does* exist on the Hub and is a public model then "
            "please ensure that it has a framework registered. If it is a "
            "private model then enable the `--use-token` flag and make "
            "sure that you are logged in to the Hub via the "
            "`huggingface-cli login` command."
        )
