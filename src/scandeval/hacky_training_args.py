"""Hacky fix to enable setting of device in TrainingArguments."""

import torch
from transformers import TrainingArguments


class TrainingArgumentsWithMPSSupport(TrainingArguments):
    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
