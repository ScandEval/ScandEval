"""Trainer subclass for question answering tasks."""

from typing import List, Optional, Union

from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalLoopOutput, PredictionOutput


class QuestionAnsweringTrainer(Trainer):
    """Trainer subclass for question answering tasks."""

    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics  # type: ignore[has-type]
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if (
            self.post_process_function is not None
            and self.compute_metrics is not None
            and self.args.should_save
        ):
            # Only the main node write the results by default
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        else:
            metrics = {}

        # Only the main node log the results by default
        if self.args.should_log:
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics  # type: ignore[has-type]
        )
        return metrics

    def predict(
        self,
        predict_dataset: Dataset,
        predict_examples: BatchEncoding,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> Union[PredictionOutput, EvalLoopOutput]:
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics  # type: ignore[has-type]
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            predict_examples, predict_dataset, output.predictions, "predict"
        )
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics,
        )
