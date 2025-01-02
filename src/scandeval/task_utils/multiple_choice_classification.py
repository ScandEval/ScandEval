"""Utility functions related to the multiple-choice classification task group."""

import logging
import typing as t

from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer, Trainer

if t.TYPE_CHECKING:
    pass


logger = logging.getLogger("scandeval")


class MultipleChoiceClassificationTrainer(Trainer):
    """Trainer subclass for question answering tasks."""

    def evaluate(
        self,
        eval_dataset: "Dataset | None" = None,
        orig_eval_dataset: "Dataset | None" = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float] | None:
        """Evaluate the model on the given dataset.

        Args:
            eval_dataset:
                The dataset to evaluate on. If None, then use the stored evaluation
                dataset.
            orig_eval_dataset:
                The original evaluation dataset, before any postprocessing. If None,
                then use the stored original evaluation dataset.
            ignore_keys:
                The keys to ignore when computing the metrics.
            metric_key_prefix:
                The prefix to use for the metric keys.

        Returns:
            The metrics computed on the evaluation dataset.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

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
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if orig_eval_dataset is not None:
            preds_and_labels = postprocess_predictions_and_labels(
                predictions=output.predictions,
                dataset=orig_eval_dataset,
                prepared_dataset=eval_dataset,
                cls_token_index=self.cls_token_id,
            )
            output.metrics.update(self.compute_metrics(preds_and_labels))

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(output.metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(
                        key
                    )

        # Only the main node log the results by default
        if self.args.should_log:
            self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
            output.metrics,
        )
        return output.metrics


# TODO: Implement this
def prepare_train_examples(
    examples: "BatchEncoding", tokenizer: "PreTrainedTokenizer"
) -> "BatchEncoding":
    """Prepare the features for training.

    Args:
        examples:
            The examples to prepare.
        tokenizer:
            The tokenizer to use to prepare the examples.

    Returns:
        The prepared examples.
    """
    raise NotImplementedError


# TODO: Implement this
def postprocess_predictions_and_labels(
    predictions: list,
    dataset: "Dataset",
    prepared_dataset: "Dataset",
    cls_token_index: int,
) -> tuple[list[dict], list[dict]]:
    """Postprocess the predictions and labels, to allow easier metric computation.

    Args:
        predictions:
            A pair of (start_logits, end_logits) predictions.
        dataset:
            The dataset containing the examples.
        prepared_dataset:
            The dataset containing the prepared examples.
        cls_token_index:
            The index of the CLS token.

    Returns:
        The postprocessed predictions and labels.
    """
    raise NotImplementedError
