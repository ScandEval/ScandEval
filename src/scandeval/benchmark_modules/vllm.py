"""Generative models using the vLLM inference framework."""

import collections.abc as c
import importlib.util
import itertools as it
import json
import logging
import random
import re
import sys
import typing as t
from functools import cached_property, partial
from pathlib import Path
from time import sleep
from types import MethodType

import torch
from datasets import DatasetDict
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, Trainer
from urllib3.exceptions import RequestError

from ..constants import MAX_LOGPROBS, SUPERTASKS_USING_LOGPROBS, TASKS_USING_JSON
from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import BatchingPreference, Framework, ModelType
from ..exceptions import (
    InvalidBenchmark,
    InvalidModel,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..languages import get_all_languages
from ..structured_generation_utils import get_ner_logits_processors
from ..task_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..types import ExtractLabelsFunction
from ..utils import clear_memory, create_model_cache_dir, get_end_of_chat_token_ids
from .hf import HuggingFaceEncoderModel, get_model_repo_info

if t.TYPE_CHECKING or importlib.util.find_spec("vllm") is not None:
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest

    try:
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )
    except ImportError:
        from vllm.distributed.parallel_state import destroy_model_parallel

if t.TYPE_CHECKING or importlib.util.find_spec("ray") is not None:
    import ray

logger = logging.getLogger("scandeval")


class VLLMModel(HuggingFaceEncoderModel):
    """A generative model using the vLLM inference framework."""

    _is_generative = True
    batching_preference = BatchingPreference.ALL_AT_ONCE

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the vLLM model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
        """
        if (
            importlib.util.find_spec("vllm") is None
            or importlib.util.find_spec("ray") is None
        ):
            raise NeedsExtraInstalled(extra="generative")

        self.model_config = model_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self.output_scores = (
            self.dataset_config.task.supertask in SUPERTASKS_USING_LOGPROBS
        )

        model, tokenizer = self._load_model_and_tokenizer()
        self._model: LLM = model
        self._tokenizer: PreTrainedTokenizer = tokenizer

        self.lora_request: LoRARequest | None = None
        if self.model_config.adapter_base_model_id is not None:
            adapter_path = snapshot_download(
                repo_id=self.model_config.model_id,
                cache_dir=Path(self.model_config.model_cache_dir),
            )
            self.lora_request = LoRARequest(
                lora_name="adapter", lora_int_id=1, lora_path=adapter_path
            )

        self._log_metadata()

    @cached_property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        match self.dataset_config.task.supertask:
            case "sequence-classification":
                return partial(
                    sequence_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case "text-to-text":
                return text_to_text.extract_labels_from_generation
            case "token-classification":
                return partial(
                    token_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case "question-answering":
                return question_answering.extract_labels_from_generation
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
        if task.supertask == "question-answering":
            dataset = dataset.map(
                lambda examples: dict(
                    label=[
                        dict(
                            id=id,
                            answers=dict(
                                answer_start=answer_dct["answer_start"],
                                text=[
                                    answer_text.lower()
                                    for answer_text in answer_dct["text"]
                                ],
                            ),
                        )
                        for id, answer_dct in zip(examples["id"], examples["answers"])
                    ]
                ),
                batched=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        if self.benchmark_config.few_shot:
            few_shot_examples = self._extract_few_shot_examples(
                dataset=dataset, task=task, itr_idx=itr_idx
            )
            dataset["test"] = dataset["test"].map(
                partial(
                    self._apply_few_shot_prompt,
                    few_shot_examples=few_shot_examples,
                    task=task,
                ),
                batched=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )
        else:
            dataset["test"] = dataset["test"].map(
                partial(self._apply_instruction_prompt, task=task),
                batched=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )
        return dataset

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        # Define which tokens to use as stopping criteria. We want to use the padding
        # token, end-of-sentence token, and a double newline (since these separate the
        # few-shot examples in the input)
        stop_tokens: list[str] = ["\n\n"]
        if self._tokenizer.pad_token_id is not None:
            stop_tokens.append(self._tokenizer.pad_token)
        if self._tokenizer.eos_token_id is not None:
            stop_tokens.append(self._tokenizer.eos_token)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                self._tokenizer.pad_token = self._tokenizer.eos_token
        if (
            self._tokenizer.bos_token_id is not None
            and self._tokenizer.pad_token_id is None
        ):
            self._tokenizer.pad_token_id = self._tokenizer.bos_token_id
            self._tokenizer.pad_token = self._tokenizer.bos_token
        assert self._tokenizer.pad_token_id is not None

        # Add end of chat token as a stopping token, if it exists
        end_of_chat_token_ids = get_end_of_chat_token_ids(tokenizer=self._tokenizer)
        if end_of_chat_token_ids is not None:
            end_of_chat_token = self._tokenizer.decode(end_of_chat_token_ids).strip()
            if end_of_chat_token:
                stop_tokens.append(end_of_chat_token)

        if self.dataset_config.task.name in TASKS_USING_JSON:
            ner_tag_names = list(self.dataset_config.prompt_label_mapping.values())
            logits_processors = get_ner_logits_processors(
                ner_tag_names=ner_tag_names, llm=self._model
            )
        else:
            logits_processors = None

        # Define the parameters used for vLLM generation
        max_tokens: int = self.dataset_config.max_generated_tokens
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            logprobs=MAX_LOGPROBS if self.output_scores else None,
            temperature=0.0,
            stop=[stop_token for stop_token in stop_tokens if stop_token],
            logits_processors=logits_processors,
        )

        # If any of the prompts are empty then we need to replace them with a BOS token
        # so that the vLLM model can generate from them
        prompts = inputs["text"]
        if any(len(prompt) == 0 for prompt in prompts):
            logger.debug("Found empty prompts, replacing with BOS token.")
            prompts = [
                prompt if len(prompt) > 0 else self._tokenizer.bos_token
                for prompt in prompts
            ]

        # Generate sequences using vLLM
        input_is_a_test = len(prompts) == 1 and len(set(prompts[0])) == 1
        raw_outputs = self._model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=(not input_is_a_test),
            lora_request=self.lora_request,
        )
        completions = self._tokenizer.batch_decode(
            sequences=[
                torch.LongTensor(output.outputs[0].token_ids) for output in raw_outputs
            ],
            skip_special_tokens=True,
        )

        # Add logprobs scores to the output
        if self.output_scores:
            scores: list[list[list[tuple[str, float]]]] = [
                [
                    [
                        (obj.decoded_token, obj.logprob)
                        for obj in token_logprobs_dict.values()
                    ]
                    for token_logprobs_dict in raw_output.outputs[0].logprobs
                ]
                for raw_output in raw_outputs
            ]
            output = GenerativeModelOutput(sequences=completions, scores=scores)
        else:
            output = GenerativeModelOutput(sequences=completions)

        return output

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
        return model_info is not None and model_info.pipeline_tag == "text-generation"

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
            model_type=ModelType.HF_HUB_GENERATIVE,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=model_info.adapter_base_model_id,
        )

        return model_config

    def _load_model_and_tokenizer(self) -> "tuple[LLM, PreTrainedTokenizer]":
        """Load the model and tokenizer.

        Returns:
            The loaded model and tokenizer.
        """
        hf_model_config = AutoConfig.from_pretrained(
            self.model_config.model_id,
            revision=self.model_config.revision,
            cache_dir=self.model_config.model_cache_dir,
        )

        quantization = None
        if hasattr(hf_model_config, "quantization_config"):
            quantization = hf_model_config.quantization_config.get("quant_method")

        # The quantised models require extra dependencies
        if quantization == "gptq" and (
            importlib.util.find_spec("auto_gptq") is None
            or importlib.util.find_spec("optimum") is None
        ):
            raise NeedsExtraInstalled(extra="quantization")
        if quantization == "awq" and importlib.util.find_spec("awq") is None:
            raise NeedsExtraInstalled(extra="quantization")

        dtype: str | torch.dtype = "auto"
        if quantization is not None and hf_model_config.torch_dtype != torch.float16:
            logger.info(
                "You are loading a quantized model with dtype "
                f"{hf_model_config.torch_dtype}, which vLLM does not support. Setting "
                "dtype to float16 instead."
            )
            dtype = torch.float16

        if self.model_config.adapter_base_model_id is not None:
            download_dir = str(Path(self.model_config.model_cache_dir) / "base_model")
        else:
            download_dir = str(self.model_config.model_cache_dir)

        max_model_len = 5_000
        potential_max_model_length_config_names = [
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
            "sliding_window",
            "sliding_window_size",
            "n_positions",
        ]
        for config_name in potential_max_model_length_config_names:
            if hasattr(hf_model_config, config_name):
                model_len = getattr(hf_model_config, config_name)
                if model_len is not None:
                    max_model_len = min(max_model_len, model_len)

        vllm_kwargs = dict(
            model=self.model_config.adapter_base_model_id or self.model_config.model_id,
            tokenizer=self.model_config.adapter_base_model_id
            or self.model_config.model_id,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
            download_dir=download_dir,
            trust_remote_code=self.benchmark_config.trust_remote_code,
            revision=self.model_config.revision,
            seed=4242,
            distributed_executor_backend="ray",
            tensor_parallel_size=torch.cuda.device_count(),
            disable_custom_all_reduce=True,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=True,
            max_logprobs=MAX_LOGPROBS if self.output_scores else None,
            # TEMP: Prefix caching isn't supported with sliding window in vLLM yet, so
            # we disable it for now
            enable_prefix_caching=False,
            enable_lora=self.model_config.adapter_base_model_id is not None,
            max_lora_rank=256,
        )

        clear_vllm()
        model = LLM(**vllm_kwargs)
        model._run_engine = MethodType(_run_engine_with_fixed_progress_bars, model)
        model.config = hf_model_config

        tokenizer = load_tokenizer(
            model_id=self.model_config.model_id,
            trust_remote_code=self.benchmark_config.trust_remote_code,
            model_max_length=max_model_len,
        )

        return model, tokenizer

    def _extract_few_shot_examples(
        self, dataset: DatasetDict, task: Task, itr_idx: int
    ) -> list[dict[str, t.Any]]:
        """Extract few-shot examples from a dataset.

        This will always extract the examples from the training split.

        We ensure that the few-shot examples are unique by picking them one at a time.

        Args:
            dataset:
                The dataset to extract the few-shot examples from.
            task:
                The task that is being benchmarked.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The few-shot examples.
        """
        random_seed = 4242 + itr_idx
        num_few_shots = self.dataset_config.num_few_shot_examples
        few_shot_examples: list[dict[str, t.Any]] = list()
        shuffled_train = dataset["train"].shuffle(seed=random_seed)

        match task.supertask:
            case "sequence-classification":
                labels = it.cycle(self.dataset_config.task.labels)
                while len(few_shot_examples) < num_few_shots:
                    label = next(labels)
                    possible_examples = shuffled_train.filter(
                        lambda x: x["label"].lower() == label.lower()
                    )
                    if len(possible_examples) == 0:
                        continue
                    example = possible_examples.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )

            case "text-to-text":
                while len(few_shot_examples) < num_few_shots:
                    example = shuffled_train.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )

            case "token-classification":
                labels = it.cycle(
                    [
                        label.lower()
                        for label in self.dataset_config.task.labels
                        if label.lower().startswith("b-")
                    ]
                )
                while len(few_shot_examples) < num_few_shots:
                    label = next(labels)
                    possible_examples = shuffled_train.filter(
                        lambda x: label in [tag.lower() for tag in x["labels"]]
                    )
                    if len(possible_examples) == 0:
                        continue
                    example = possible_examples.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )

            case "question-answering":
                # Locate the maximum number of tokens that constitutes a short example
                for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
                    train_with_short_examples = dataset["train"].filter(
                        lambda example: len(example["context"]) < max_num_tokens
                    )
                    num_short_examples = len(train_with_short_examples)
                    if num_short_examples >= self.dataset_config.num_few_shot_examples:
                        break
                else:
                    raise InvalidBenchmark(
                        "Could not find enough short examples for few-shot learning."
                    )

                shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
                while len(few_shot_examples) < num_few_shots:
                    example = shuffled_train.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["context"] != example["context"]
                    )

            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {task.supertask}."
                )

        random.seed(random_seed)
        random.shuffle(few_shot_examples)
        return few_shot_examples

    # TODO: Change to \n\n separation instead of ChatML
    def _apply_few_shot_prompt(
        self,
        examples: dict[str, t.Any],
        few_shot_examples: list[dict[str, t.Any]],
        task: Task,
    ) -> dict[str, t.Any]:
        """Apply few-shot examples to an example.

        Args:
            examples:
                The examples to apply the few-shot examples to.
            few_shot_examples:
                The few-shot examples to apply.
            task:
                The task that is being benchmarked.

        Returns:
            The example with the few-shot examples applied.
        """

        def split_section(section: str) -> tuple[str, str]:
            """Split a section of the prompt to user and assistant messages."""
            user_part = "\n".join(section.split("\n")[:-1])
            assistant_part = section.split("\n")[-1]
            return user_part, assistant_part

        few_shot_messages: list[str]
        match task.supertask:
            case "sequence-classification" | "text-to-text":
                label_name = (
                    "label"
                    if task.supertask == "sequence-classification"
                    else "target_text"
                )
                few_shot_messages = [
                    self.dataset_config.prompt_template.format(
                        text=example["text"].replace("\n", " ").strip(),
                        label=example[label_name].strip(),
                    )
                    for example in few_shot_examples
                ]

                prompt_prefix = ""
                if self.dataset_config.prompt_prefix:
                    prompt_prefix = self.dataset_config.prompt_prefix + "\n\n"

                examples["text"] = [
                    prompt_prefix
                    + "\n\n".join(few_shot_messages)
                    + "\n\n"
                    + self.dataset_config.prompt_template.format(text=text, label="")
                    for text in examples["text"]
                ]

            # case "token-classification":

            #     def create_label(example: dict) -> str:
            #         prompt_labels = self.dataset_config.prompt_label_mapping.values()
            #         labels: dict[str, list[str]] = {
            #             prompt_label: list() for prompt_label in prompt_labels
            #         }
            #         for token, label in zip(example["tokens"], example["labels"]):
            #             label = label.lower()
            #             if label == "o":
            #                 continue
            #             prompt_label = self.dataset_config.prompt_label_mapping[label]
            #             if label.startswith("b-"):
            #                 labels[prompt_label].append(token)
            #             elif label.startswith("i-"):
            #                 labels[prompt_label][-1] += " " + token
            #         return json.dumps(labels, ensure_ascii=False)

            #     prompt_sections = [
            #         self.dataset_config.prompt_template.format(
            #             text=" ".join(example["tokens"]).replace("\n", " ").strip(),
            #             label=create_label(example=example),
            #         )
            #         for example in few_shot_examples
            #     ]

            #     few_shot_messages = [
            #         dict(role=role, content=content.split(":", 1)[1].strip())
            #         for section in prompt_sections[1:]
            #         for role, content in zip(
            #             it.cycle(["user", "assistant"]), split_section(section=section)
            #         )
            #         if content.split(":", 1)[1].strip() != ""
            #     ]

            #     if self.dataset_config.prompt_prefix:
            #         few_shot_messages[0]["content"] = (
            #             self.dataset_config.prompt_prefix
            #             + "\n\n"
            #             + few_shot_messages[0]["content"]
            #         )

            #     examples["messages"] = [
            #         few_shot_messages
            #         + [
            #             dict(
            #                 role="user",
            #                 content=split_section(
            #                     section=self.dataset_config.prompt_template.format(
            #                         text=" ".join(tokens).replace("\n", " ").strip(),
            #                         label="",
            #                     )
            #                 )[0]
            #                 .split(":", 1)[1]
            #                 .strip(),
            #             )
            #         ]
            #         for tokens in examples["tokens"]
            #     ]

            # case "question-answering":
            #     prompt_sections = [
            #         self.dataset_config.prompt_template.format(
            #             text=example["context"].replace("\n", " ").strip(),
            #             question=example["question"].strip(),
            #             label=example["answers"]["text"][0],
            #         )
            #         for example in few_shot_examples
            #     ]

            #     few_shot_messages = [
            #         dict(role=role, content=content.split(":", 1)[1].strip())
            #         for section in prompt_sections[1:]
            #         for role, content in zip(
            #             it.cycle(["user", "assistant"]), split_section(section=section)
            #         )
            #         if content.split(":", 1)[1].strip() != ""
            #     ]

            #     if self.dataset_config.prompt_prefix:
            #         few_shot_messages[0]["content"] = (
            #             self.dataset_config.prompt_prefix
            #             + "\n\n"
            #             + few_shot_messages[0]["content"]
            #         )

            #     examples["messages"] = [
            #         few_shot_messages
            #         + [
            #             dict(
            #                 role="user",
            #                 content=split_section(
            #                     section=self.dataset_config.prompt_template.format(
            #                         text=context.replace("\n", " ").strip(),
            #                         question=question,
            #                         label="",
            #                     )
            #                 )[0]
            #                 .split(":", 1)[1]
            #                 .strip(),
            #             )
            #         ]
            #         for context, question in zip(
            #             examples["context"], examples["question"]
            #         )
            #     ]

            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {task.supertask}."
                )

        assert len(
            {len(values) for values in examples.values()}
        ), "The number of examples and messages must be the same."
        return examples

    def _apply_instruction_prompt(
        self, examples: dict[str, t.Any], task: Task
    ) -> dict[str, t.Any]:
        """Apply instruction prompts to an example.

        Args:
            examples:
                The examples to apply the instruction prompts to.
            task:
                The task that is being benchmarked.

        Returns:
            The example with the instruction prompts applied.
        """
        match task.supertask:
            case "sequence-classification" | "text-to-text" | "token-classification":
                prompts = [
                    self.dataset_config.instruction_prompt.format(
                        text=re.sub(r"\n+", "\n", text).strip()
                    )
                    for text in examples["text"]
                ]

            case "question-answering":
                prompts = [
                    self.dataset_config.instruction_prompt.format(
                        text=re.sub(pattern=r"\n+", repl="\n", string=text).strip(),
                        question=question.strip(),
                    )
                    for (text, question) in zip(
                        examples["context"], examples["question"]
                    )
                ]

            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {task.supertask}."
                )

        examples["text"] = prompts
        return examples

    def __del__(self) -> None:
        """Clear the GPU memory used by the model, and remove the model itself."""
        if hasattr(self, "_model"):
            del self._model
        del self
        clear_vllm()

    @cached_property
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for vLLM models."
        )

    @cached_property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for vLLM models."
        )


def load_tokenizer(
    model_id: str, trust_remote_code: bool, model_max_length: int
) -> "PreTrainedTokenizer":
    """Load the tokenizer.

    Args:
        model_id:
            The model identifier. Used for logging.
        trust_remote_code:
            Whether to trust remote code.
        model_max_length:
            The maximum length of the model.

    Returns:
        The loaded tokenizer.
    """
    while True:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                verbose=False,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                truncation_side="left",
                model_max_length=model_max_length,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except (json.JSONDecodeError, OSError, TypeError):
            raise InvalidModel(f"Could not load tokenizer for model {model_id!r}.")
        except (TimeoutError, RequestError):
            logger.info(f"Couldn't load tokenizer for {model_id!r}. Retrying.")
            sleep(5)
            continue


def _run_engine_with_fixed_progress_bars(
    self: "LLM", use_tqdm: bool
) -> list["RequestOutput"]:
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(
            total=num_requests, leave=False, disable=hasattr(sys, "_called_from_test")
        )
    else:
        pbar = None

    # Run the engine.
    outputs: list["RequestOutput"] = list()
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Sort the outputs by request ID. This is necessary because some requests may be
    # finished earlier than its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))

    return outputs


def clear_vllm() -> None:
    """Clear the GPU memory used by the vLLM model, enabling re-initialisation."""
    if importlib.util.find_spec("vllm") is not None:
        try:
            destroy_model_parallel()
        except ImportError:
            pass
        clear_memory()
        ray.shutdown()
