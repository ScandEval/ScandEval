"""Unit tests for the `callbacks` module."""

from collections.abc import Sized
from dataclasses import dataclass

import pytest

from src.scandeval.callbacks import NeverLeaveProgressCallback


@dataclass
class FakeState:
    is_local_process_zero: bool = True


@dataclass
class FakeEvalDataloader:
    dataset: Sized = (1, 2, 3)

    def __len__(self):
        return len(self.dataset)


class TestNeverLeaveProgressCallback:
    @pytest.fixture(scope="class")
    def callback(self):
        yield NeverLeaveProgressCallback()

    @pytest.fixture(scope="class")
    def state(self):
        yield FakeState(is_local_process_zero=True)

    @pytest.fixture(scope="class")
    def eval_dataloader(self):
        yield FakeEvalDataloader()

    def test_on_train_begin_initialises_not_leaving_pbar(self, callback, state):
        assert callback.training_bar is None
        callback.on_train_begin(args=None, state=state, control=None)
        assert not callback.training_bar.leave

    def test_on_prediction_step_initialises_not_leaving_pbar(
        self, callback, state, eval_dataloader
    ):
        assert callback.prediction_bar is None
        callback.on_prediction_step(
            args=None, state=state, control=None, eval_dataloader=eval_dataloader
        )
        assert not callback.prediction_bar.leave
