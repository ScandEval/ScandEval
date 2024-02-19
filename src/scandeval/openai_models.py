"""Model and tokenizer wrapper for OpenAI models."""

import logging

import torch
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding, GenerationConfig, PretrainedConfig
from transformers.modeling_utils import ModelOutput

from .config import BenchmarkConfig, ModelConfig
from .exceptions import NeedsExtraInstalled
from .types import is_list_of_int, is_list_of_list_of_int, is_list_of_str

try:
    import tiktoken
    from openai import NotFoundError, OpenAI
except ImportError:
    OpenAI = None
    tiktoken = None

logger = logging.getLogger(__package__)


class OpenAITokenizer:
    """An OpenAI tokenizer.

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
        """Initialize the tokenizer.

        Args:
            model_config:
                The model configuration.
            hf_model_config:
                The Hugging Face model configuration.
        """
        self.model_config = model_config
        self.hf_model_config = hf_model_config

        self.bos_token_id: int = self.hf_model_config.bos_token_id or -1
        self.cls_token_id: int = self.bos_token_id
        self.eos_token_id: int = self.hf_model_config.eos_token_id or -1
        self.sep_token_id: int = self.eos_token_id
        self.pad_token_id: int = self.hf_model_config.pad_token_id or -1

        self.bos_token = self.encoding.decode([self.bos_token_id])
        self.cls_token = self.bos_token
        self.eos_token = self.encoding.decode([self.eos_token_id])
        self.sep_token = self.eos_token

    @property
    def encoding(self) -> "tiktoken.Encoding":
        """Return the underlying tiktoken encoding."""
        return tiktoken.encoding_for_model(model_name=self.model_config.model_id)

    def __call__(self, text: str | list[str], **kwargs) -> BatchEncoding:
        """Tokenize text.

        Args:
            text:
                The text to tokenize.
            **kwargs:
                Additional keyword arguments.

        Returns:
            The tokenized text.
        """
        truncation = kwargs.get("truncation", False)
        start_idx = -self.model_max_length if truncation else 0

        text_list = [text] if isinstance(text, str) else text
        encoded_inputs = [
            BatchEncoding(
                dict(
                    input_ids=self.encoding.encode(
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

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode token IDs.

        Args:
            token_ids:
                The token IDs to decode.
            **kwargs:
                Additional keyword arguments.

        Returns:
            The decoded text.
        """
        token_ids = [
            token_id for token_id in token_ids if token_id != self.pad_token_id
        ]
        return self.encoding.decode(tokens=token_ids)

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        """Decode batched token IDs.

        Args:
            sequences:
                The token IDs to decode.
            **kwargs:
                Additional keyword arguments.

        Returns:
            The decoded text.
        """
        return [self.decode(token_ids=sequence) for sequence in sequences]

    def encode(self, text: str | list[str] | list[int], **kwargs) -> list[int]:
        """Encode text.

        Args:
            text:
                The text to encode.
            **kwargs:
                Additional keyword arguments.

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
            skip_special_tokens:
                Whether to skip special tokens. Defaults to False.

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
        encoded_inputs: (
            BatchEncoding
            | list[BatchEncoding]
            | dict[str, list[int]]
            | dict[str, list[list[int]]]
            | list[dict[str, list[int]]]
        ),
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
                Additional keyword arguments.
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

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return self.encoding.max_token_value + 1


class OpenAIModel:
    """An OpenAI model.

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
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hf_model_config: PretrainedConfig,
        benchmark_config: BenchmarkConfig,
        tokenizer: OpenAITokenizer,
    ) -> None:
        """Initialize the model.

        Args:
            model_config:
                The model configuration.
            hf_model_config:
                The Hugging Face model configuration.
            benchmark_config:
                The benchmark configuration.
            tokenizer:
                The tokenizer.
        """
        if OpenAI is None:
            raise NeedsExtraInstalled(extra="openai")

        self.model_config = model_config
        self.config = hf_model_config
        self.benchmark_config = benchmark_config
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.client = OpenAI(
            api_key=self.benchmark_config.openai_api_key, max_retries=60
        )
        self.is_chat_model = self._is_chat_model()

    def _is_chat_model(self) -> bool:
        """Returns whether the model is a chat model."""
        try:
            self.client.completions.create(
                model=self.model_config.model_id, prompt="Test", max_tokens=1
            )
            return False
        except NotFoundError as e:
            if "This is a chat model" in str(e):
                return True
            raise e

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
        temperature = (
            0.0 if not generation_config.do_sample else generation_config.temperature
        )
        generation_kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=generation_config.top_p,
            n=generation_config.num_return_sequences,
            frequency_penalty=generation_config.repetition_penalty - 1.0,
            stop=["\n\n", self.tokenizer.eos_token, self.tokenizer.pad_token],
        )

        if not self.is_chat_model:
            model_output = self.client.completions.create(
                prompt=prompt, **generation_kwargs
            )
            generation_output = model_output.choices[0].text.strip()
        else:
            model_output = self.client.chat.completions.create(
                messages=[dict(role="user", content=prompt)], **generation_kwargs
            )
            generation_output = model_output.choices[0].message.content.strip()

        completion_ids = self.tokenizer([generation_output]).input_ids.tolist()
        output = LongTensor(completion_ids)

        if generation_config.return_dict_in_generate:
            output = ModelOutput(dict(sequences=output))

        return output
