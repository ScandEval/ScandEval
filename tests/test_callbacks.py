"""Unit tests for the `callbacks` module."""

from collections.abc import Sized
from dataclasses import dataclass
from typing import Generator

import pytest

from scandeval.callbacks import NeverLeaveProgressCallback


@dataclass
class FakeState:
    """Dummy state class for testing."""

    is_local_process_zero: bool = True


@dataclass
class FakeEvalDataloader:
    """Dummy evaluation dataloader class for testing."""

    dataset: Sized = (1, 2, 3)

    def __len__(self):
        """Return the length of the dataloader."""
        return len(self.dataset)


class TestNeverLeaveProgressCallback:
    """Tests for the `NeverLeaveProgressCallback` class."""

    @pytest.fixture(scope="class")
    def callback(self) -> Generator[NeverLeaveProgressCallback, None, None]:
        """Yields a callback."""
        yield NeverLeaveProgressCallback()

    @pytest.fixture(scope="class")
    def state(self) -> Generator[FakeState, None, None]:
        """Yields a state."""
        yield FakeState(is_local_process_zero=True)

    @pytest.fixture(scope="class")
    def eval_dataloader(self):
        """Yields an evaluation dataloader."""
        yield FakeEvalDataloader()

    def test_on_train_begin_initialises_not_leaving_pbar(self, callback, state):
        """Test that the `leave` attribute on the training progress bar is False."""
        assert callback.training_bar is None
        callback.on_train_begin(args=None, state=state, control=None)
        assert callback.training_bar is not None
        assert not callback.training_bar.leave

    def test_on_prediction_step_initialises_not_leaving_pbar(
        self, callback, state, eval_dataloader
    ):
        """Test that the `leave` attribute on the prediction progress bar is False."""
        assert callback.prediction_bar is None
        callback.on_prediction_step(
            args=None, state=state, control=None, eval_dataloader=eval_dataloader
        )
        assert callback.prediction_bar is not None
        assert not callback.prediction_bar.leave
