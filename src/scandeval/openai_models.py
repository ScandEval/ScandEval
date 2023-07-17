"""Model and tokenizer wrapper for OpenAI models."""

import json
import logging
from pathlib import Path
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
from transformers import BatchEncoding, GenerationConfig, PretrainedConfig
from transformers.modeling_utils import ModelOutput

from .config import BenchmarkConfig, ModelConfig
from .exceptions import InvalidBenchmark
from .types import is_list_of_int, is_list_of_list_of_int, is_list_of_str

logger = logging.getLogger(__package__)


class OpenAITokenizer:
    """An OpenAI tokenizer.

    Args:
        model_config:
            The model configuration.
        hf_model_config:
            The Hugging Face model configuration.

    Attributes:
        model_config:
            The model configuration.
        hf_model_config:
            The Hugging Face model configuration.
        encoding:
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
            text:
                The text to tokenize.

        Returns:
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
            token_ids:
                The token IDs to decode.

        Returns:
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
            text:
                The text to encode.

        Returns:
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
            The maximum length of a sequence for this model.
        """
        return self.hf_model_config.model_max_length

    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = False
    ) -> str | list[str]:
        """Convert token IDs to tokens.

        Args:
            ids:
                The token IDs to convert.

        Returns:
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
            tokens:
                The tokens to convert.

        Returns:
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
            encoded_inputs:
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
        model_config:
            The model configuration.
        hf_model_config:
            The Hugging Face model configuration.
        benchmark_config:
            The benchmark configuration.
        tokenizer:
            The tokenizer.

    Attributes:
        model_config:
            The model configuration.
        config:
            The Hugging Face model configuration.
        benchmark_config:
            The benchmark configuration.
        tokenizer:
            The tokenizer.
        device:
            The device to use, is always CPU.
        is_chat_model:
            Whether the model is a chat model.
        cache_path:
            The path to the cache.
        cache:
            A cache for the model's outputs.
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
        self.is_chat_model = self._is_chat_model()

        model_cache_dir = Path(model_config.model_cache_dir)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = model_cache_dir / "model_outputs.json"
        if not self.cache_path.exists():
            with self.cache_path.open("w") as f:
                json.dump(dict(), f)
        with self.cache_path.open() as f:
            self.cache: dict[str, dict[str, str]] = json.load(f)

    def _is_chat_model(self) -> bool:
        """Returns whether the model is a chat model."""
        for _ in range(60):
            try:
                openai.Completion.create(
                    model=self.model_config.model_id, prompt="Test", max_tokens=1
                )
                return False
            except InvalidRequestError as e:
                if "This is a chat model" in str(e):
                    return True
                else:
                    raise e
            except (RateLimitError, ServiceUnavailableError, APIError, Timeout):
                sleep(1)
        else:
            raise InvalidBenchmark("OpenAI API is not available")

    def generate(
        self,
        inputs: Tensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> Tensor | LongTensor | ModelOutput:
        """Generate text using the model.

        Args:
            inputs:
                The input IDs, of shape (batch_size, sequence_length).
            generation_config:
                The generation configuration. If None then a default GenerationConfig
                will be used. Defaults to None.
            **generation_kwargs:
                Additional keyword arguments. Can also be used to override
                generation configuration.

        Returns:
            The model output.
        """
        if generation_config is None:
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            for key, value in generation_kwargs.items():
                setattr(generation_config, key, value)

        multiple_inputs = inputs.size(dim=0) > 1
        if multiple_inputs:
            raise ValueError(
                "OpenAI models do not support multiple inputs. Please use a batch "
                "size of 1."
            )

        two_dimensional_input = len(inputs.size()) == 2
        if two_dimensional_input:
            inputs = inputs[0]

        prompt = self.tokenizer.decode(
            [
                token_id
                for token_id in inputs.tolist()
                if token_id != self.config.pad_token_id
            ]
        )

        model_id = self.model_config.model_id
        max_tokens: int = generation_config.max_new_tokens or 1
        generation_kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
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

        max_tokens_str = str(max_tokens)
        if max_tokens_str not in self.cache:
            self.cache[max_tokens_str] = dict()

        # Use cache if possible
        if prompt in self.cache[max_tokens_str]:
            generation_output = self.cache[max_tokens_str][prompt]
        else:
            for _ in range(60):
                try:
                    if not self.is_chat_model:
                        model_output = openai.Completion.create(
                            prompt=prompt, **generation_kwargs
                        )
                        generation_output = model_output.choices[0].text.strip()
                    else:
                        model_output = openai.ChatCompletion.create(
                            messages=[dict(role="user", content=prompt)],
                            **generation_kwargs,
                        )
                        generation_output = model_output.choices[
                            0
                        ].message.content.strip()
                    break
                except (RateLimitError, ServiceUnavailableError, APIError, Timeout):
                    sleep(1)
            else:
                raise InvalidBenchmark("OpenAI API is not available")

            self.cache[max_tokens_str][prompt] = generation_output
            with self.cache_path.open("w") as f:
                json.dump(self.cache, f, indent=4)

        completion_ids = self.tokenizer([generation_output]).input_ids.tolist()
        output = LongTensor(completion_ids)

        if generation_config.return_dict_in_generate:
            output = ModelOutput(dict(sequences=output))

        return output
