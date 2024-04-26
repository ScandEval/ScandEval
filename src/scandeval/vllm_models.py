"""A wrapper for vLLM models."""

import importlib.util
import logging
import math
import sys
from types import MethodType
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.utils import ModelOutput

from .exceptions import NeedsExtraInstalled
from .structured_generation_utils import get_ner_logits_processors
from .tasks import NER
from .utils import clear_memory, get_end_of_chat_token_ids

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig, PreTrainedTokenizerBase
    from vllm import LLM, RequestOutput

    from .config import DatasetConfig, ModelConfig

if importlib.util.find_spec("ray") is not None:
    import ray

if importlib.util.find_spec("vllm") is not None:
    from vllm import LLM, SamplingParams

    try:
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )
    except ImportError:
        from vllm.distributed.parallel_state import destroy_model_parallel


logger = logging.getLogger(__package__)


class VLLMModel:
    """A wrapper for vLLM models."""

    def __init__(
        self,
        model_config: "ModelConfig",
        hf_model_config: "PretrainedConfig",
        dataset_config: "DatasetConfig",
        model_cache_dir: "str | Path",
        trust_remote_code: bool,
        tokenizer: "PreTrainedTokenizerBase | None" = None,
    ) -> None:
        """Initialize a vLLM model.

        Args:
            model_config:
                A model configuration.
            hf_model_config:
                A Hugging Face model configuration.
            dataset_config:
                A dataset configuration.
            model_cache_dir:
                The directory to cache the model in.
            trust_remote_code:
                Whether to trust remote code, e.g., from Hugging Face.
            tokenizer:
                A Hugging Face tokenizer. If None, the tokenizer will need to be
                loaded separately.
        """
        self.model_config = model_config
        self.config = hf_model_config
        self.dataset_config = dataset_config
        self.model_cache_dir = model_cache_dir
        self.trust_remote_code = trust_remote_code
        self.device = torch.device("cuda")
        self.tokenizer = tokenizer

        # This is required to be able to re-initialize the model, in case we have
        # already initialized it once
        destroy_model_parallel()
        clear_memory()
        ray.shutdown()

        self.max_model_len = 5_000
        potential_max_model_length_config_names = [
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
            "n_positions",
        ]
        for config_name in potential_max_model_length_config_names:
            if hasattr(hf_model_config, config_name):
                self.max_model_len = min(
                    self.max_model_len, getattr(hf_model_config, config_name)
                )

        quantization = None
        if hasattr(self.config, "quantization_config"):
            quantization = self.config.quantization_config.get("quant_method", None)

        # The quantised models require extra dependencies
        if quantization == "gptq" and (
            importlib.util.find_spec("auto_gptq") is None
            or importlib.util.find_spec("optimum") is None
        ):
            raise NeedsExtraInstalled(extra="quantization")
        if quantization == "awq" and importlib.util.find_spec("awq") is None:
            raise NeedsExtraInstalled(extra="quantization")

        dtype = "auto"
        if quantization is not None and self.config.torch_dtype != torch.float16:
            logger.info(
                "You are loading a quantized model with dtype "
                f"{self.config.torch_dtype}, which vLLM does not support. Setting "
                "dtype to float16 instead."
            )
            dtype = torch.float16

        vllm_kwargs = dict(
            model=self.model_config.model_id,
            gpu_memory_utilization=0.95,
            max_model_len=self.max_model_len,
            download_dir=str(self.model_cache_dir),
            trust_remote_code=self.trust_remote_code,
            revision=self.model_config.revision,
            seed=4242,
            tensor_parallel_size=torch.cuda.device_count(),
            disable_custom_all_reduce=True,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=True,
            max_logprobs=10,
            enable_prefix_caching=True,
        )

        while True:
            try:
                self._model = LLM(**vllm_kwargs)
                break
            except NotImplementedError as e:
                if "prefix caching" in str(e):
                    vllm_kwargs.pop("enable_prefix_caching")
                    self._model = LLM(**vllm_kwargs)
                    continue
                raise e

        self._model._run_engine = MethodType(
            _run_engine_with_fixed_progress_bars, self._model
        )

    def __del__(self) -> None:
        """Clear the GPU memory used by the model, and remove the model itself."""
        destroy_model_parallel()
        if hasattr(self, "_model"):
            del self._model
        del self
        clear_memory()
        ray.shutdown()

    def generate(
        self,
        inputs: torch.Tensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> torch.Tensor | torch.LongTensor | ModelOutput:
        """Generate sequences using the model.

        Args:
            inputs:
                The input batch of sequences to generate from.
            generation_config:
                The generation config to use for generation.
            **generation_kwargs:
                Additional generation kwargs to pass to the model.

        Returns:
            The generated sequences.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded to generate sequences.")

        if generation_config is None:
            generation_config = GenerationConfig(**generation_kwargs)
        else:
            for key, value in generation_kwargs.items():
                setattr(generation_config, key, value)

        # Define which tokens to use as stopping criteria. We want to use the padding
        # token, end-of-sentence token, and a double newline (since these separate the
        # few-shot examples in the input)
        stop_tokens: list[str] = ["\n\n"]
        if self.tokenizer.pad_token_id is not None:
            stop_tokens.append(self.tokenizer.pad_token)
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.append(self.tokenizer.eos_token)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            self.tokenizer.bos_token_id is not None
            and self.tokenizer.pad_token_id is None
        ):
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            self.tokenizer.pad_token = self.tokenizer.bos_token
        assert self.tokenizer.pad_token_id is not None

        # Add end of chat token as a stopping token, if it exists
        end_of_chat_token_ids = get_end_of_chat_token_ids(tokenizer=self.tokenizer)
        if end_of_chat_token_ids is not None:
            end_of_chat_token = self.tokenizer.decode(end_of_chat_token_ids).strip()
            stop_tokens.append(end_of_chat_token)

        # Define the parameters used for vLLM generation
        max_tokens: int = generation_config.max_new_tokens or 1
        temperature = (
            0.0 if not generation_config.do_sample else generation_config.temperature
        )
        sampling_params = SamplingParams(
            # What to output
            max_tokens=max_tokens,
            logprobs=10 if generation_config.output_scores else None,
            n=generation_config.num_return_sequences,
            # How to sample
            temperature=temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            stop=stop_tokens,
            repetition_penalty=generation_config.repetition_penalty,
            frequency_penalty=generation_config.repetition_penalty - 1.0,
            logits_processors=self.logits_processors,
        )

        # The inputs are tokenised, so we decode them to get the original text, which
        # is the input to the vLLM model
        prompts = self.tokenizer.batch_decode(
            sequences=inputs, skip_special_tokens=True
        )

        # If any of the prompts are empty then we need to replace them with a BOS token
        # so that the vLLM model can generate from them
        if any(len(prompt) == 0 for prompt in prompts):
            logger.debug("Found empty prompts, replacing with BOS token.")
            prompts = [
                prompt if len(prompt) > 0 else self.tokenizer.bos_token
                for prompt in prompts
            ]

        # Generate sequences using vLLM
        input_is_a_test = len(prompts) == 1 and len(set(prompts[0])) == 1
        raw_outputs = self._model.generate(
            prompts=prompts,
            use_tqdm=(not input_is_a_test),
            sampling_params=sampling_params,
        )

        # Collect the generated sequences into a single tensor of shape
        # (batch_size, generated_sequence_length)
        output = torch.nn.utils.rnn.pad_sequence(
            sequences=[
                torch.LongTensor(output.outputs[0].token_ids) for output in raw_outputs
            ],
            batch_first=True,
            padding_value=float(self.tokenizer.pad_token_id),
        )

        if generation_config.return_dict_in_generate:
            # Add logprobs scores to the output
            if generation_config.output_scores:
                # Create a list with placeholder logprobs for every token generated.
                # Each tensor in the list will be of shape (batch_size, vocab_size)
                batch_size = len(raw_outputs)
                vocab_size = len(self.tokenizer.get_vocab())
                max_seq_len = max(
                    len(raw_output.outputs[0].logprobs) for raw_output in raw_outputs
                )
                scores = [
                    torch.full(size=(batch_size, vocab_size), fill_value=-math.inf)
                    for _ in range(max_seq_len)
                ]

                # Fill in the logprobs for each generated token. The logprobs from the
                # vLLM output only contain the logprobs for the top-k tokens, so we
                # only fill in these and leave the rest at ~0% probability
                for sample_idx, raw_output in enumerate(raw_outputs):
                    assert raw_output.outputs[0].logprobs is not None
                    seq_len = len(raw_output.outputs[0].logprobs)
                    for gen_token_idx in range(seq_len):
                        logprobs_dict = raw_output.outputs[0].logprobs[gen_token_idx]
                        for token_idx, logprob_obj in logprobs_dict.items():
                            logprob = logprob_obj.logprob
                            scores[gen_token_idx][sample_idx, token_idx] = logprob

                output = ModelOutput(dict(sequences=output, scores=tuple(scores)))
            else:
                output = ModelOutput(dict(sequences=output))

        return output

    def __call__(
        self,
        inputs: torch.Tensor,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> torch.Tensor | torch.LongTensor | ModelOutput:
        """Generate sequences using the model.

        Args:
            inputs:
                The input batch of sequences to generate from.
            generation_config:
                The generation config to use for generation.
            **generation_kwargs:
                Additional generation kwargs to pass to the model.

        Returns:
            The generated sequences.
        """
        return self.generate(
            inputs=inputs, generation_config=generation_config, **generation_kwargs
        )

    def build_logits_processors(self) -> None:
        """Return the logits processors to use for structured generation.

        This requires the model and tokenizer to be set.

        Raises:
            ValueError:
                If the model or tokenizer is not set.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set to build logits processors.")

        logits_processors = list()

        if self.dataset_config.task == NER:
            ner_tag_names = list(self.dataset_config.prompt_label_mapping.values())
            ner_logits_processors = get_ner_logits_processors(
                ner_tag_names=ner_tag_names, llm=self._model
            )
            logits_processors.extend(ner_logits_processors)

        self.logits_processors = logits_processors

    def set_tokenizer(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        """Set the tokenizer to use for generation.

        Args:
            tokenizer:
                The tokenizer to use for generation.
        """
        self.tokenizer = tokenizer
        self._model.set_tokenizer(tokenizer)

    def to(self, _: torch.device) -> None:
        """Dummy method to make the model compatible with the benchmarking script."""
        pass

    def eval(self) -> None:
        """Dummy method to make the model compatible with the benchmarking script."""
        pass

    def children(self) -> list:
        """Dummy method to make the model compatible with the benchmarking script."""
        return []


def _run_engine_with_fixed_progress_bars(
    self: "LLM", use_tqdm: bool
) -> list["RequestOutput"]:
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(
            total=num_requests, leave=False, disable=hasattr(sys, "_called_from_test")
        )

    # Run the engine.
    outputs: list["RequestOutput"] = list()
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    pbar.update(1)

    if use_tqdm:
        pbar.close()

    # Sort the outputs by request ID. This is necessary because some requests may be
    # finished earlier than its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))

    return outputs
