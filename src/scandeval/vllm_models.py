"""A wrapper for vLLM models."""

import logging
import warnings
from pathlib import Path
from types import MethodType

import torch
from tqdm import tqdm
from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer
from transformers.utils import ModelOutput
from vllm import LLM, RequestOutput, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from .config import ModelConfig
from .utils import HiddenPrints, clear_memory

logger = logging.getLogger(__package__)


class VLLMModel:
    """A wrapper for vLLM models."""

    def __init__(
        self,
        model_config: ModelConfig,
        hf_model_config: PretrainedConfig,
        model_cache_dir: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> None:
        """Initialize a vLLM model.

        Args:
            model_config:
                A model configuration.
            hf_model_config:
                A Hugging Face model configuration.
            model_cache_dir:
                The directory to cache the model in.
            tokenizer:
                A Hugging Face tokenizer. If None, the tokenizer will need to be
                loaded separately.
        """
        self.model_config = model_config
        self.config = hf_model_config
        self.device = torch.device("cuda")
        self.tokenizer = tokenizer
        with warnings.catch_warnings(), HiddenPrints():
            warnings.simplefilter("ignore", category=UserWarning)

            # This is required to be able to re-initialize the model, in case we
            # have already initialized it once
            destroy_model_parallel()
            clear_memory()

            max_model_len = 10_000
            if hasattr(hf_model_config, "max_position_embeddings"):
                max_model_len = min(10_000, hf_model_config.max_position_embeddings)
            if hasattr(hf_model_config, "model_max_length"):
                max_model_len = min(10_000, hf_model_config.model_max_length)
            if hasattr(hf_model_config, "n_positions"):
                max_model_len = min(10_000, hf_model_config.n_positions)

            self._model = LLM(
                model=model_config.model_id,
                gpu_memory_utilization=0.9,
                max_model_len=max_model_len,
                download_dir=str(model_cache_dir),
                trust_remote_code=True,
            )
            self._model._run_engine = MethodType(
                _run_engine_with_fixed_progress_bars, self._model
            )

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
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            n=generation_config.num_return_sequences,
            repetition_penalty=generation_config.repetition_penalty,
            frequency_penalty=generation_config.repetition_penalty - 1.0,
            stop=stop_tokens,
            logprobs=10 if generation_config.output_scores else None,
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
        input_is_a_test = len(prompts) == 1 and len(prompts[0]) == 1
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
                    torch.full(size=(batch_size, vocab_size), fill_value=-1e9)
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
            inputs=inputs,
            generation_config=generation_config,
            **generation_kwargs,
        )

    def to(self, _: torch.device) -> None:
        """Dummy method to make the model compatible with the benchmarking script."""
        pass

    def eval(self) -> None:
        """Dummy method to make the model compatible with the benchmarking script."""
        pass

    def children(self) -> list:
        """Dummy method to make the model compatible with the benchmarking script."""
        return []


def _run_engine_with_fixed_progress_bars(self, use_tqdm: bool) -> list[RequestOutput]:
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, leave=False)

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
