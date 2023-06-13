"""Model and tokenizer wrapper for OpenAI models."""

import openai
import tiktoken
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding, GenerationConfig, PretrainedConfig
from transformers.utils import ModelOutput

from .config import BenchmarkConfig, DatasetConfig, ModelConfig


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

    def __init__(
        self, model_config: ModelConfig, hf_model_config: PretrainedConfig
    ) -> None:
        self.model_config = model_config
        self.hf_model_config = hf_model_config
        self.encoding = tiktoken.encoding_for_model(model_name=model_config.model_id)

        self.bos_token_id: int = self.hf_model_config.bos_token_id
        self.cls_token_id: int = self.bos_token_id
        self.eos_token_id: int = self.hf_model_config.eos_token_id
        self.sep_token_id: int = self.eos_token_id
        self.pad_token_id: int = self.hf_model_config.pad_token_id

        self.bos_token = self.encoding.decode([self.bos_token_id])
        self.cls_token = self.bos_token
        self.eos_token = self.encoding.decode([self.eos_token_id])
        self.sep_token = self.eos_token
        self.pad_token = "<pad>"

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

    def generate(
        self,
        inputs: Tensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> ModelOutput | LongTensor:
        """Generate text using the model.

        Args:
            inputs (LongTensor):
                The input IDs.
            generation_config (GenerationConfig or None, optional):
                The generation configuration. If None then a default GenerationConfig
                will be used. Defaults to None.
            **generation_kwargs:
                Additional keyword arguments. Can also be used to override
                generation configuration.

        Returns:
            ModelOutput or LongTensor:
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

        generation_output = openai.Completion.create(
            model=self.model_config.model_id,
            prompt=inputs_list,
            max_tokens=generation_config.max_length,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            n=generation_config.num_return_sequences,
            frequency_penalty=generation_config.repetition_penalty - 1.0,
            stop=["\n\n", self.tokenizer.eos_token, self.tokenizer.pad_token],
        )

        completion_ids_list: list[list[int]] = [
            self.tokenizer(choice.text.strip())["input_ids"][0].tolist()
            for choice in generation_output.choices
        ]

        if multiple_inputs:
            return LongTensor(
                pad_sequence(
                    sequences=list(map(Tensor, completion_ids_list)),
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
            )
        else:
            return LongTensor(completion_ids_list)
