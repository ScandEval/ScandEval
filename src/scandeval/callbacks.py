"""Callbacks for the Hugging Face Trainer."""

import sys
from collections.abc import Sized

from tqdm.auto import tqdm
from transformers.trainer_callback import ProgressCallback


class NeverLeaveProgressCallback(ProgressCallback):
    """Progress callback which never leaves the progress bar."""

    def __init__(self, *args, **kwargs):
        """Initialise the callback."""
        super().__init__(*args, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        """Callback actions when training begins."""
        if state.is_local_process_zero:
            desc = "Finetuning model"
            self.training_bar = tqdm(
                total=None,
                leave=False,
                desc=desc,
                disable=hasattr(sys, "_called_from_test"),
            )
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Callback actions when a training step ends."""
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        """Callback actions when a prediction step ends."""
        if eval_dataloader is None:
            return
        correct_dtype = isinstance(eval_dataloader.dataset, Sized)
        if state.is_local_process_zero and correct_dtype:
            if self.prediction_bar is None:
                desc = "Evaluating model"
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader),
                    leave=False,
                    desc=desc,
                    disable=hasattr(sys, "_called_from_test"),
                )
            self.prediction_bar.update(1)
