"""Generative models from an inference API, using the LiteLLM framework."""

import collections.abc as c
import itertools as it
import json
import logging
import random
import re
import typing as t
from functools import cached_property, partial
from time import sleep

import litellm
from datasets import DatasetDict
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
)
from litellm.types.utils import ModelResponse
from transformers import Trainer

from ..constants import MAX_LOGPROBS, SUPERTASKS_USING_LOGPROBS, TASKS_USING_JSON
from ..data_models import BenchmarkConfig, GenerativeModelOutput, ModelConfig, Task
from ..enums import BatchingPreference, Framework, ModelType
from ..exceptions import (
    InvalidBenchmark,
    NeedsAdditionalArgument,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..task_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..types import ExtractLabelsFunction
from ..utils import create_model_cache_dir
from .base import BenchmarkModule

logger = logging.getLogger("scandeval")


VOCAB_SIZE_MAPPING = {
    # OpenAI models
    "(text-)?(ada|babbage|curie|davinci)(-001)?": 50_257,
    "(code|text)-davinci-00[2-9]": 50_281,
    "gpt-3.5-turbo(-16k)?(-[0-9]{4})?": 100_256,
    "gpt-4-(32k)?(-[0-9]{4})?": 100_256,
    "gpt-4-[0-9]{4}-preview": 100_256,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 100_256,
    "gpt-4-(vision|turbo)(-preview)?": 100_256,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": 100_256,
    "gpt-4o(-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_019,
    # Anthropic models
    "claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": -1,
}


MODEL_MAX_LENGTH_MAPPING = {
    # OpenAI models
    "(text-)?(ada|babbage|curie|davinci)(-001)?": 2_050,
    "text-davinci-00[2-9]": 4_098,
    "code-davinci-00[1-9]": 8_002,
    "gpt-3.5-turbo-0613": 4_096,
    "gpt-3.5-turbo(-[0-9]{4})?": 16_385,
    "gpt-3.5-turbo-16k(-[0-9]{4})?": 16_384,
    "gpt-4(-[0-9]{4})?": 8_191,
    "gpt-4-32k(-[0-9]{4})?": 32_767,
    "gpt-4-[0-9]{4}-preview": 128_000,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    "gpt-4-(vision|turbo)(-preview)?": 128_000,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": 4_095,
    "gpt-4o(-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    # Anthropic models
    "claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": 200_000,
}


NUM_PARAMS_MAPPING = {
    "(text-)?ada(-001)?": 350_000_000,
    "(text-)?babbage(-001)?": 3_000_000_000,
    "(text-)?curie(-001)?": 13_000_000_000,
    "((text|code)-)?davinci(-00[1-9])?": 175_000_000_000,
    "gpt-(3.5|4)-turbo-((16|32)k)?(-[0-9]{4})?": -1,
    "gpt-4-[0-9]{4}-preview": -1,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    "gpt-4-(vision|turbo)(-preview)?": -1,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": -1,
    "gpt-4o(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    "gpt-4o-mini(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    # Anthropic models
    "claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": -1,
}


class LiteLLMModel(BenchmarkModule):
    """A generative model from LiteLLM."""

    _is_generative = True
    batching_preference = BatchingPreference.SINGLE_SAMPLE

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        assert "messages" in inputs, "The input must contain a 'messages' key."
        assert (
            len(inputs["messages"]) == 1
        ), "API models only support single-sample batching."
        messages = inputs["messages"][0]

        generation_kwargs: dict[str, t.Any] = dict(
            model=self.model_config.model_id,
            max_tokens=self.dataset_config.max_generated_tokens,
            stop=["\n\n"],
            temperature=0.0,
            seed=4242,
        )

        if self.dataset_config.task.supertask in SUPERTASKS_USING_LOGPROBS:
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = MAX_LOGPROBS

        if self.dataset_config.task.name in TASKS_USING_JSON:
            assert (
                "json" in messages[0]["content"]
            ), "Prompt must contain 'json' for JSON tasks."
            generation_kwargs["response_format"] = dict(type="json_object")

        # This drops generation kwargs that are not supported by the model
        litellm.drop_params = True

        # Extract the generated sequences from the model response. Some APIs cannot
        # handle using newlines as stop sequences, so we try both.
        num_attempts = 10
        for _ in range(num_attempts):
            try:
                model_response = litellm.completion(
                    messages=messages, max_retries=3, **generation_kwargs
                )
                break
            except BadRequestError as e:
                if "stop_sequences" not in str(e).lower():
                    raise InvalidBenchmark(
                        f"Failed to generate text. The error message was: {e}"
                    )
                generation_kwargs["stop"] = None
            except APIError as e:
                raise InvalidBenchmark(
                    f"Failed to generate text. The error message was: {e}"
                )
            except InternalServerError as e:
                if "overloaded" not in str(e).lower():
                    raise InvalidBenchmark(
                        f"Failed to generate text. The error message was: {e}"
                    )
                sleep(1)
                continue
            except AuthenticationError:
                raise NeedsAdditionalArgument(
                    cli_argument="--api-key",
                    script_argument="api_key=<your-api-key>",
                    run_with_cli=self.benchmark_config.run_with_cli,
                )
        else:
            raise InvalidBenchmark(
                message=f"Failed to generate text, after {num_attempts} attempts."
            )

        assert isinstance(model_response, ModelResponse)
        model_response_choices = model_response.choices[0]
        assert isinstance(model_response_choices, litellm.Choices)
        generation_output = model_response_choices.message["content"].strip()

        # Structure the model output as a GenerativeModelOutput object
        model_output = GenerativeModelOutput(sequences=[generation_output])
        if hasattr(model_response_choices, "logprobs"):
            logprobs_list: list[list[tuple[str, float]]] = [
                [(dct["token"], dct["logprob"]) for dct in content["top_logprobs"]]
                for content in model_response_choices.logprobs["content"]
            ]
            model_output.scores = [logprobs_list]

        return model_output

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        for key, value in NUM_PARAMS_MAPPING.items():
            if re.match(pattern=key, string=self.model_config.model_id) is not None:
                return value
        return -1

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        for key, value in VOCAB_SIZE_MAPPING.items():
            if re.match(pattern=key, string=self.model_config.model_id) is not None:
                return value
        return -1

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        for key, value in MODEL_MAX_LENGTH_MAPPING.items():
            if re.match(pattern=key, string=self.model_config.model_id) is not None:
                return value
        return -1

    @cached_property
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for LiteLLM models."
        )

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

    @cached_property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for LiteLLM models."
        )

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
        if model_id in litellm.model_list:
            return True

        try:
            litellm.completion(
                messages=[dict(role="user", content="X")], model=model_id, max_tokens=1
            )
            return True
        except (BadRequestError, NotFoundError):
            candidate_models = [
                candidate_model_id
                for candidate_model_id in litellm.model_list
                if candidate_model_id.startswith(model_id)
            ]
            match len(candidate_models):
                case 0:
                    pass
                case 1:
                    logger.warning(
                        f"Could not find the model ID {model_id!r}. Did you mean "
                        f"{candidate_models[0]!r}?"
                    )
                case _:
                    candidate_models_str = "', '".join(candidate_models)
                    logger.warning(
                        f"Could not find the model ID {model_id!r}. Did you mean any of "
                        f"the following model IDs: '{candidate_models_str}'?"
                    )
            return False

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
        return ModelConfig(
            model_id=model_id,
            revision="main",
            framework=Framework.API,
            task="text-generation",
            languages=list(),
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            model_type=ModelType.API,
            adapter_base_model_id=None,
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
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(self._apply_prompt, few_shot_examples=few_shot_examples, task=task),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

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
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
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
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
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
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
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
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
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

    def _apply_prompt(
        self,
        examples: dict[str, t.Any],
        few_shot_examples: list[dict[str, t.Any]],
        task: Task,
    ) -> dict[str, t.Any]:
        """Apply prompt template to an example, potentially with few-shot examples.

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

        match task.supertask:
            case "sequence-classification":
                few_shot_sections = [
                    self.dataset_config.prompt_template.format(
                        text=example["text"].replace("\n", " ").strip(),
                        label=example["label"].replace("\n", " ").strip(),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    self.dataset_config.prompt_template.format(
                        text=text.replace("\n", " ").strip(), label=""
                    )
                    for text in examples["text"]
                ]

            case "text-to-text":
                few_shot_sections = [
                    self.dataset_config.prompt_template.format(
                        text=example["text"].replace("\n", " ").strip(),
                        target_text=example["target_text"].replace("\n", " ").strip(),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    self.dataset_config.prompt_template.format(
                        text=text.replace("\n", " ").strip(), target_text=""
                    )
                    for text in examples["text"]
                ]

            case "token-classification":

                def create_label(example: dict) -> str:
                    prompt_labels = self.dataset_config.prompt_label_mapping.values()
                    labels: dict[str, list[str]] = {
                        prompt_label: list() for prompt_label in prompt_labels
                    }
                    for token, label in zip(example["tokens"], example["labels"]):
                        label = label.lower()
                        if label == "o":
                            continue
                        prompt_label = self.dataset_config.prompt_label_mapping[label]
                        if label.startswith("b-"):
                            labels[prompt_label].append(token)
                        elif label.startswith("i-"):
                            labels[prompt_label][-1] += " " + token
                    return json.dumps(labels, ensure_ascii=False)

                few_shot_sections = [
                    self.dataset_config.prompt_template.format(
                        text=" ".join(example["tokens"]).replace("\n", " ").strip(),
                        label=create_label(example=example),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    self.dataset_config.prompt_template.format(
                        text=" ".join(tokens).replace("\n", " ").strip(), label=""
                    )
                    for tokens in examples["tokens"]
                ]

            case "question-answering":
                few_shot_sections = [
                    self.dataset_config.prompt_template.format(
                        text=example["context"].replace("\n", " ").strip(),
                        question=example["question"].replace("\n", " ").strip(),
                        label=example["answers"]["text"][0].replace("\n", " "),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    self.dataset_config.prompt_template.format(
                        text=context.replace("\n", " ").strip(),
                        question=question.replace("\n", " ").strip(),
                        label="",
                    )
                    for context, question in zip(
                        examples["context"], examples["question"]
                    )
                ]

            case _:
                raise NotImplementedError(
                    f"Unsupported task supertask: {task.supertask}."
                )

        few_shot_messages = [
            dict(role=role, content=content.split(":", 1)[1].strip())
            for section in few_shot_sections
            for role, content in zip(
                it.cycle(["user", "assistant"]), split_section(section=section)
            )
            if content.split(":", 1)[1].strip() != ""
        ]

        if self.dataset_config.prompt_prefix:
            few_shot_messages[0]["content"] = (
                self.dataset_config.prompt_prefix
                + "\n\n"
                + few_shot_messages[0]["content"]
            )

        examples["messages"] = [
            few_shot_messages
            + [
                dict(
                    role="user",
                    content=split_section(section=new_section)[0]
                    .split(":", 1)[1]
                    .strip(),
                )
            ]
            for new_section in new_sections
        ]

        return examples
