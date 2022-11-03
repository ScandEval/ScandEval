"""Callbacks for the Hugging Face Trainer."""

from collections.abc import Sized

from tqdm.auto import tqdm
from transformers.trainer_callback import ProgressCallback


class NeverLeaveProgressCallback(ProgressCallback):
    """Progress callback which never leaves the progress bar"""

    def __init__(self, *args, testing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.testing = testing

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            desc = "Finetuning model"
            self.training_bar = tqdm(
                total=None,
                leave=False,
                desc=desc,
                disable=self.testing,
            )
        self.current_step = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
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
                    disable=self.testing,
                )
            self.prediction_bar.update(1)
