"""Model and tokenizer wrapper for OpenAI models."""

import logging
from time import sleep

import openai
import tiktoken
import torch
from openai.error import APIError, InvalidRequestError, RateLimitError
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import BatchEncoding, GenerationConfig, PretrainedConfig

from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .types import is_list_of_int, is_list_of_str

logger = logging.getLogger(__name__)


class OpenAITokenizer:
    """An OpenAI tokenizer.

    Args:
        model_config (ModelConfig):
            The model configuration.
        hf_model_config (PretrainedConfig):
            The Hugging Face model configuration.

    Attributes:
        model_config (ModelConfig):
            The model configuration.
        hf_model_config (PretrainedConfig):
            The Hugging Face model configuration.
        encoding (Encoding):
            The encoding.
    """

    unk_token = "<unk>"
    unk_token_id = -1
    pad_token = "<pad>"
    padding_side = "left"
    is_fast = False

    def __init__(
        self, model_config: ModelConfig, hf_model_config: PretrainedConfig
    ) -> None:
        self.model_config = model_config
        self.hf_model_config = hf_model_config
        self.encoding = tiktoken.encoding_for_model(model_name=model_config.model_id)

        self.bos_token_id: int = self.hf_model_config.bos_token_id or -1
        self.cls_token_id: int = self.bos_token_id
        self.eos_token_id: int = self.hf_model_config.eos_token_id or -1
        self.sep_token_id: int = self.eos_token_id
        self.pad_token_id: int = self.hf_model_config.pad_token_id or -1

        self.bos_token = self.encoding.decode([self.bos_token_id])
        self.cls_token = self.bos_token
        self.eos_token = self.encoding.decode([self.eos_token_id])
        self.sep_token = self.eos_token

    def __call__(self, text: str | list[str], **kwargs) -> BatchEncoding:
        """Tokenize text.

        Args:
            text (str):
                The text to tokenize.

        Returns:
            dict[str, LongTensor]:
                The tokenized text.
        """
        text_list = [text] if isinstance(text, str) else text
        input_ids = [
            Tensor(
                self.encoding.encode(
                    text,
                    allowed_special={
                        self.bos_token,
                        self.eos_token,
                        self.cls_token,
                        self.sep_token,
                        self.pad_token,
                    },
                )
            )
            for text in text_list
        ]
        padded_input_ids = pad_sequence(
            sequences=input_ids, batch_first=True, padding_value=self.pad_token_id
        ).long()
        return BatchEncoding(dict(input_ids=padded_input_ids))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs.

        Args:
            token_ids (list of int):
                The token IDs to decode.

        Returns:
            str:
                The decoded text.
        """
        token_ids = [
            token_id for token_id in token_ids if token_id != self.pad_token_id
        ]
        return self.encoding.decode(tokens=token_ids)

    def encode(self, text: str | list[str] | list[int], **kwargs) -> list[int]:
        """Encode text.

        Args:
            text (str or list of str or list of int):
                The text to encode.

        Returns:
            list of int:
                The encoded text.
        """
        if is_list_of_int(text):
            return text
        elif is_list_of_str(text) or isinstance(text, str):
            return self(text, **kwargs).input_ids.tolist()
        else:
            raise TypeError(f"Cannot encode {type(text)}.")

    @property
    def special_tokens_map(self) -> dict[str, str | list[str]]:
        """A mapping of special tokens to their values.

        Returns:
            dict[str, str or list of str]:
                The mapping of special tokens to their values.
        """
        return dict(
            bos_token=self.bos_token,
            cls_token=self.cls_token,
            eos_token=self.eos_token,
            sep_token=self.sep_token,
            pad_token=self.pad_token,
        )

    @property
    def model_max_length(self) -> int:
        """The maximum length of a sequence for this model.

        Returns:
            int:
                The maximum length of a sequence for this model.
        """
        return self.hf_model_config.model_max_length

    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = False
    ) -> str | list[str]:
        """Convert token IDs to tokens.

        Args:
            ids (int or list of int):

        Returns:
            str or list of str:
                The tokens.
        """
        if isinstance(ids, int):
            ids = [ids]
        tokens = self.encoding.decode(tokens=ids)
        if skip_special_tokens:
            tokens = [
                token
                for token in tokens
                if token not in self.special_tokens_map.values()
            ]
        if len(tokens) == 1:
            return tokens[0]
        return tokens

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """Convert tokens to token IDs.

        Args:
            tokens (str or list of str):
                The tokens to convert.

        Returns:
            int or list of int:
                The token IDs.
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = [self.encode(text=token)[0] for token in tokens]
        if len(ids) == 1:
            return ids[0]
        return ids


class OpenAIModel:
    """An OpenAI model.

    Args:
        model_config (ModelConfig):
            The model configuration.
        hf_model_config (PretrainedConfig):
            The Hugging Face model configuration.
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
        tokenizer (OpenAITokenizer):
            The tokenizer.

    Attributes:
        model_config (ModelConfig):
            The model configuration.
        config (PretrainedConfig):
            The Hugging Face model configuration.
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
        tokenizer (OpenAITokenizer):
            The tokenizer.
        device (torch.device):
            The device to use, is always CPU.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hf_model_config: PretrainedConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        tokenizer: OpenAITokenizer,
    ) -> None:
        self.model_config = model_config
        self.config = hf_model_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")

    def generate(
        self,
        inputs: Tensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> LongTensor:
        """Generate text using the model.

        Args:
            inputs (Tensor or list of Tensor):
                The input IDs.
            generation_config (GenerationConfig or None, optional):
                The generation configuration. If None then a default GenerationConfig
                will be used. Defaults to None.
            **generation_kwargs:
                Additional keyword arguments. Can also be used to override
                generation configuration.

        Returns:
            LongTensor:
                The model output.
        """
        if generation_config is None:
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            for key, value in generation_kwargs.items():
                setattr(generation_config, key, value)

        multiple_inputs = inputs.dim() == 2

        # Remove padding tokens
        if multiple_inputs:
            inputs_list = [
                [
                    token_id
                    for token_id in token_ids
                    if token_id != self.config.pad_token_id
                ]
                for token_ids in inputs.tolist()
            ]
        else:
            inputs_list = [
                token_id
                for token_id in inputs.tolist()
                if token_id != self.config.pad_token_id
            ]

        # Check if the model is a chat model
        try:
            openai.Completion.create(
                model=self.model_config.model_id,
                prompt="Test",
                max_tokens=10,
            )
            is_chat_model = False
        except InvalidRequestError as e:
            if "This is a chat model" in str(e):
                is_chat_model = True
            else:
                raise e

        while True:
            try:
                completion_ids_list: list[list[int]]
                if not is_chat_model:
                    generation_output = openai.Completion.create(
                        model=self.model_config.model_id,
                        prompt=inputs_list,
                        max_tokens=generation_config.max_length,
                        temperature=generation_config.temperature,
                        top_p=generation_config.top_p,
                        n=generation_config.num_return_sequences,
                        frequency_penalty=generation_config.repetition_penalty - 1.0,
                        stop=[
                            "\n\n",
                            self.tokenizer.eos_token,
                            self.tokenizer.pad_token,
                        ],
                    )
                    completion_ids_list = [
                        self.tokenizer(choice.text.strip())["input_ids"][0].tolist()
                        for choice in generation_output.choices
                    ]

                else:
                    completion_ids_list = list()
                    for input_ids in tqdm(inputs_list, leave=False):
                        single_output = openai.ChatCompletion.create(
                            model=self.model_config.model_id,
                            messages=[
                                dict(
                                    role="user",
                                    content=self.tokenizer.decode(input_ids),
                                ),
                            ],
                            max_tokens=generation_config.max_length,
                            temperature=generation_config.temperature,
                            top_p=generation_config.top_p,
                            n=generation_config.num_return_sequences,
                            frequency_penalty=(
                                generation_config.repetition_penalty - 1.0
                            ),
                            stop=[
                                "\n\n",
                                self.tokenizer.eos_token,
                                self.tokenizer.pad_token,
                            ],
                        )
                        completion_ids: list[int] = self.tokenizer(
                            single_output.choices[0].message.content.strip()
                        )["input_ids"][0].tolist()
                        completion_ids_list.append(completion_ids)
                    break

            except RateLimitError:
                logger.debug("Rate limit exceeded, trying again in a few seconds...")
                sleep(10)
                continue

            except APIError:
                logger.debug(
                    "Encountered an error with the OpenAI API, trying again in a "
                    "few seconds..."
                )
                sleep(10)
                continue

        if multiple_inputs:
            padded = pad_sequence(
                sequences=list(map(Tensor, completion_ids_list)),
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            return padded.long()
        else:
            return LongTensor(completion_ids_list)
