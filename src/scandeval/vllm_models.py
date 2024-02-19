"""A wrapper for vLLM models."""

import logging
import sys
import warnings
from pathlib import Path
from types import MethodType
from typing import Callable

import torch
from tqdm import tqdm
from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizerBase
from transformers.utils import ModelOutput

from .config import DatasetConfig, ModelConfig
from .tasks import NER
from .utils import clear_memory, get_ner_parser

logger = logging.getLogger(__package__)

try:
    from lmformatenforcer.integrations.vllm import (
        build_vllm_logits_processor,
        build_vllm_token_enforcer_tokenizer_data,
    )
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
except ImportError:
    logger.debug("Failed to import vLLM, assuming that it is not needed.")

    class LLM:  # type: ignore[no-redef]
        """Dummy class."""

        pass

    class RequestOutput:  # type: ignore[no-redef]
        """Dummy class."""

        pass

    def destroy_model_parallel():  #  type: ignore[no-redef]
        """Dummy function."""
        pass


class VLLMModel:
    """A wrapper for vLLM models."""

    def __init__(
        self,
        model_config: ModelConfig,
        hf_model_config: PretrainedConfig,
        dataset_config: DatasetConfig,
        model_cache_dir: str | Path,
        trust_remote_code: bool,
        tokenizer: PreTrainedTokenizerBase | None = None,
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
        self.device = torch.device("cuda")
        self.tokenizer = tokenizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            # This is required to be able to re-initialize the model, in case we
            # have already initialized it once
            destroy_model_parallel()
            clear_memory()

            max_model_len = 10_000
            potential_max_model_length_config_names = [
                "max_position_embeddings",
                "max_sequence_length",
                "model_max_length",
                "n_positions",
            ]
            for config_name in potential_max_model_length_config_names:
                if hasattr(hf_model_config, config_name):
                    max_model_len = min(
                        max_model_len, getattr(hf_model_config, config_name)
                    )

            self._model = LLM(
                model=model_config.model_id,
                gpu_memory_utilization=0.9,
                max_model_len=max_model_len,
                download_dir=str(model_cache_dir),
                trust_remote_code=trust_remote_code,
                revision=self.model_config.revision,
                seed=4242,
            )
            self._model._run_engine = MethodType(
                _run_engine_with_fixed_progress_bars, self._model
            )

            # Temporary fix until this vLLM PR is part of a release (should be any
            # release after 0.3.0):
            # https://github.com/vllm-project/vllm/pull/2741
            self._model.get_tokenizer = MethodType(_get_tokenizer, self._model)
            self._model.set_tokenizer = MethodType(_set_tokenizer, self._model)

    def __del__(self) -> None:
        """Clear the GPU memory used by the model, and remove the model itself."""
        destroy_model_parallel()
        if hasattr(self, "_model"):
            del self._model
        del self
        clear_memory()

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
            logits_processors=self.get_logits_processors(),
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
                    torch.full(size=(batch_size, vocab_size), fill_value=-1e3)
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
                        for token_idx, logprob in logprobs_dict.items():
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

    def get_logits_processors(self) -> list[Callable] | None:
        """Return the logits processors to use for structured generation."""
        logits_processors = list()

        # Add JSON generation constraint if we are benchmarking the NER task
        if self.dataset_config.task == NER:
            parser = get_ner_parser(dataset_config=self.dataset_config)
            tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm=self._model)
            logits_processor = build_vllm_logits_processor(
                llm=tokenizer_data, character_level_parser=parser
            )
            logits_processors.append(logits_processor)

            assert self.tokenizer is not None
            forbidden_token_ids = list()
            forbidden_tokens = ["\n", "\n\n", "\n\n\n", "\t", "\t\t", "\t\t\t"]
            for forbidden_token in forbidden_tokens:
                forbidden_token_ids.extend(
                    list(
                        self.tokenizer(
                            forbidden_token, add_special_tokens=False
                        ).input_ids
                    )
                )
            forbidden_token_ids = list(set(forbidden_token_ids))

            def no_tabs_or_newlines(_: list[int], scores: torch.Tensor) -> torch.Tensor:
                mask = torch.zeros_like(scores)
                for forbidden_token_id in forbidden_token_ids:
                    mask[forbidden_token_id] = -1e3
                return scores + mask

            logits_processors.append(no_tabs_or_newlines)

        return logits_processors

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Set the tokenizer to use for generation.

        Args:
            tokenizer:
                The tokenizer to use for generation.
        """
        self.tokenizer = tokenizer
        self._model.set_tokenizer(tokenizer)

        # TODO: Remove this block if it's not needed
        # This sets the internal tokenizer in the vLLM model. The
        # `LLM.llm_engine.tokenizer` is a `TokenizerGroup` object, which has a
        # `tokenizer` attribute that is the actual tokenizer. This is a new change from
        # `vllm` version 0.3.0, which is a breaking change since the `TokenizerGroup`
        # doesn't have the same properties and methods as a Hugging Face
        # `PreTrainedTokenizer` object. To resolve this, we copy all properties and
        # methods from the `PreTrainedTokenizer` object to the `TokenizerGroup` object,
        # unless the property or method already exists in the `TokenizerGroup` object.
        # vLLM issue on this: https://github.com/vllm-project/vllm/issues/2713
        # self._model.llm_engine.tokenizer.tokenizer = tokenizer
        # for attr in dir(tokenizer):
        #     if attr.startswith("_") or hasattr(self._model.llm_engine.tokenizer, attr):
        #         continue
        #     setattr(self._model.llm_engine.tokenizer, attr, getattr(tokenizer, attr))

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
    self: LLM, use_tqdm: bool
) -> list[RequestOutput]:
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(
            total=num_requests, leave=False, disable=hasattr(sys, "_called_from_test")
        )

    # Run the engine.
    outputs: list[RequestOutput] = list()
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


def _get_tokenizer(self: LLM) -> PreTrainedTokenizerBase:
    return self.llm_engine.tokenizer.tokenizer


def _set_tokenizer(self: LLM, tokenizer: PreTrainedTokenizerBase) -> None:
    self.llm_engine.tokenizer.tokenizer = tokenizer
