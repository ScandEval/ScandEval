"""Model and tokenizer wrapper for OpenAI models."""

import logging
from time import sleep

import openai
import tiktoken
import torch
from openai.error import (
    APIError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import BatchEncoding, GenerationConfig, PretrainedConfig
from transformers.modeling_utils import ModelOutput

from .config import BenchmarkConfig, ModelConfig
from .types import is_list_of_int, is_list_of_list_of_int, is_list_of_str

logger = logging.getLogger(__package__)


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

        self.bos_token_id: int = self.hf_model_config.bos_token_id or -1
        self.cls_token_id: int = self.bos_token_id
        self.eos_token_id: int = self.hf_model_config.eos_token_id or -1
        self.sep_token_id: int = self.eos_token_id
        self.pad_token_id: int = self.hf_model_config.pad_token_id or -1

        encoding = tiktoken.encoding_for_model(model_name=model_config.model_id)
        self.bos_token = encoding.decode([self.bos_token_id])
        self.cls_token = self.bos_token
        self.eos_token = encoding.decode([self.eos_token_id])
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
        truncation = kwargs.get("truncation", False)
        start_idx = -self.model_max_length if truncation else 0

        encoding = tiktoken.encoding_for_model(model_name=self.model_config.model_id)
        text_list = [text] if isinstance(text, str) else text
        encoded_inputs = [
            BatchEncoding(
                dict(
                    input_ids=encoding.encode(
                        text,
                        allowed_special={
                            self.bos_token,
                            self.eos_token,
                            self.cls_token,
                            self.sep_token,
                            self.pad_token,
                        },
                    )[start_idx:]
                )
            )
            for text in text_list
        ]
        return self.pad(encoded_inputs=encoded_inputs)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs.

        Args:
            token_ids (list of int):
                The token IDs to decode.

        Returns:
            str:
                The decoded text.
        """
        encoding = tiktoken.encoding_for_model(model_name=self.model_config.model_id)
        token_ids = [
            token_id for token_id in token_ids if token_id != self.pad_token_id
        ]
        return encoding.decode(tokens=token_ids)

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
        ids_list = [ids] if isinstance(ids, int) else ids
        tokens = [self.decode(token_ids=[i]) for i in ids_list]
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

    def pad(
        self,
        encoded_inputs: BatchEncoding
        | list[BatchEncoding]
        | dict[str, list[int]]
        | dict[str, list[list[int]]]
        | list[dict[str, list[int]]],
        **kwargs,
    ) -> BatchEncoding:
        """Pad encoded inputs.

        Args:
            encoded_inputs (BatchEncoding, list or dict):
                Tokenized inputs. Can represent one input (BatchEncoding or Dict[str,
                List[int]]) or a batch of tokenized inputs (list of BatchEncoding,
                Dict[str, List[List[int]]] or List[Dict[str, List[int]]]) so you can
                use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.
            **kwargs:
        """
        # Single example
        if isinstance(encoded_inputs, BatchEncoding):
            return encoded_inputs
        elif isinstance(encoded_inputs, dict) and is_list_of_int(
            encoded_inputs["input_ids"]
        ):
            return BatchEncoding(data=encoded_inputs)

        # Batch of examples
        if isinstance(encoded_inputs, dict) and is_list_of_list_of_int(
            encoded_inputs["input_ids"]
        ):
            input_ids = encoded_inputs["input_ids"]
        else:
            assert isinstance(encoded_inputs, list)
            input_ids = [list(example["input_ids"]) for example in encoded_inputs]

        # Flip the token IDs in the lists, since `pad_sequence` pads to the right by
        # default, and we want padding to the left
        flipped_input_ids: list[Tensor] = []
        for input_id_list in input_ids:
            input_id_list.reverse()
            flipped_input_ids.append(LongTensor(input_id_list))

        padded_input_ids = (
            pad_sequence(
                sequences=flipped_input_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            .flip(dims=[1])
            .long()
        )

        return BatchEncoding(dict(input_ids=padded_input_ids))


class OpenAIModel:
    """An OpenAI model.

    Args:
        model_config (ModelConfig):
            The model configuration.
        hf_model_config (PretrainedConfig):
            The Hugging Face model configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
        tokenizer (OpenAITokenizer):
            The tokenizer.

    Attributes:
        model_config (ModelConfig):
            The model configuration.
        config (PretrainedConfig):
            The Hugging Face model configuration.
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
        benchmark_config: BenchmarkConfig,
        tokenizer: OpenAITokenizer,
    ) -> None:
        self.model_config = model_config
        self.config = hf_model_config
        self.benchmark_config = benchmark_config
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")

        # Determine whether the model is a chat model
        while True:
            try:
                openai.Completion.create(
                    model=self.model_config.model_id,
                    prompt="Test",
                    max_tokens=1,
                )
                self.is_chat_model = False
                break
            except InvalidRequestError as e:
                if "This is a chat model" in str(e):
                    self.is_chat_model = True
                    break
                else:
                    raise e
            except (RateLimitError, ServiceUnavailableError, APIError, Timeout):
                sleep(1)
                continue

    # TODO: Consider caching the generations. This will remove a small amount of noise
    # of course, but it will reduce the amount of API calls drastically
    def generate(
        self,
        inputs: Tensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> Tensor | LongTensor | ModelOutput:
        """Generate text using the model.

        Args:
            inputs (Tensor):
                The input IDs.
            generation_config (GenerationConfig or None, optional):
                The generation configuration. If None then a default GenerationConfig
                will be used. Defaults to None.
            **generation_kwargs:
                Additional keyword arguments. Can also be used to override
                generation configuration.

        Returns:
            Tensor or ModelOutput:
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

        completion_ids_list: list[list[int]]
        if not self.is_chat_model:
            while True:
                try:
                    generation_output = openai.Completion.create(
                        model=self.model_config.model_id,
                        prompt=inputs_list,
                        max_tokens=generation_config.max_new_tokens,
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
                    break
                except (RateLimitError, ServiceUnavailableError, APIError, Timeout):
                    sleep(1)

        else:
            completion_ids_list = list()
            for input_ids in tqdm(inputs_list, leave=False):
                while True:
                    try:
                        single_output = openai.ChatCompletion.create(
                            model=self.model_config.model_id,
                            messages=[
                                dict(
                                    role="user",
                                    content=self.tokenizer.decode(input_ids),
                                ),
                            ],
                            max_tokens=generation_config.max_new_tokens,
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

                    except (RateLimitError, ServiceUnavailableError, APIError, Timeout):
                        sleep(1)

        if multiple_inputs:
            output = pad_sequence(
                sequences=list(map(Tensor, completion_ids_list)),
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).long()
        else:
            output = LongTensor(completion_ids_list)

        if generation_config.return_dict_in_generate:
            output = ModelOutput(dict(sequences=output))

        return output
