"""Question answering Trainer subclass."""

from collections import defaultdict

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.trainer import Trainer

from .utils import get_special_token_metadata


class QuestionAnsweringTrainer(Trainer):
    """Trainer subclass for question answering tasks."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the trainer."""
        super().__init__(*args, **kwargs)

        # Get the CLS token id for the tokenizer
        special_token_metadata = get_special_token_metadata(self.tokenizer)
        self.cls_token_id = special_token_metadata["cls_token_id"]

        # Set the label names
        self.label_names = ["start_positions", "end_positions"]

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        orig_eval_dataset: Dataset | None = None,
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


def postprocess_predictions_and_labels(
    predictions: list, dataset: Dataset, prepared_dataset: Dataset, cls_token_index: int
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
    # Extract the logits from the predictions
    all_start_logits = predictions[0]
    all_end_logits = predictions[1]

    # Build a map from an example to its corresponding features, being the blocks of
    # text from the context that we're feeding into the model. An example can have
    # multiple features/blocks if it has a long context.
    id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(prepared_dataset):
        id = feature["id"]
        example_index = id_to_index[id]
        features_per_example[example_index].append(i)

    # Loop over all the examples
    predictions = list()
    labels = list()
    for example_index, example in enumerate(dataset):
        # Extract the best valid answer associated with the current example
        best_answer = find_best_answer(
            all_start_logits=all_start_logits,
            all_end_logits=all_end_logits,
            prepared_dataset=prepared_dataset,
            feature_indices=features_per_example[example_index],
            context=example["context"],
            max_answer_length=30,
            num_best_logits=20,
            min_null_score=0.0,
            cls_token_index=cls_token_index,
        )

        # Create the final prediction dictionary, to be added to the list of
        # predictions
        prediction = dict(
            id=example["id"], prediction_text=best_answer, no_answer_probability=0.0
        )

        # Add the answer to the list of predictions
        predictions.append(prediction)

        # Create the associated reference dictionary, to be added to the list of
        # references
        label = dict(
            id=example["id"],
            answers=dict(
                text=example["answers"]["text"],
                answer_start=example["answers"]["answer_start"],
            ),
        )

        # Add the answer and label to the list of predictions and labels, respectively
        labels.append(label)

    return predictions, labels


def find_best_answer(
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    prepared_dataset: Dataset,
    feature_indices: list[int],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
    cls_token_index: int,
) -> str:
    """Find the best answer for a given example.

    Args:
        all_start_logits:
            The start logits for all the features.
        all_end_logits:
            The end logits for all the features.
        prepared_dataset:
            The dataset containing the prepared examples.
        feature_indices:
            The indices of the features associated with the current example.
        context:
            The context of the example.
        max_answer_length:
            The maximum length of the answer.
        num_best_logits:
            The number of best logits to consider.
        min_null_score:
            The minimum score an answer can have.
        cls_token_index:
            The index of the CLS token.

    Returns:
        The best answer for the example.
    """
    # Loop through all the features associated to the current example
    valid_answers = list()
    for feature_index in feature_indices:
        # Get the features associated with the current example
        features = prepared_dataset[feature_index]

        # Get the predictions of the model for this feature
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]

        # Update minimum null prediction
        cls_index = features["input_ids"].index(cls_token_index)
        feature_null_score = (start_logits[cls_index] + end_logits[cls_index]).item()
        if min_null_score < feature_null_score:
            min_null_score = feature_null_score

        # Find the valid answers for the feature
        valid_answers_for_feature = find_valid_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=features["offset_mapping"],
            context=context,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        valid_answers.extend(valid_answers_for_feature)

    # In the very rare edge case we have not a single non-null prediction, we create a
    # fake prediction to avoid failure
    if not valid_answers:
        return ""

    # Otherwise, we select the answer with the largest score as the best answer, and
    # return it
    best_answer_dict = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    return best_answer_dict["text"]


def find_valid_answers(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: list[tuple[int, int]],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
) -> list[dict]:
    """Find the valid answers from the start and end indexes.

    Args:
        start_logits:
            The logits for the start of the answer.
        end_logits:
            The logits for the end of the answer.
        offset_mapping:
            The offset mapping, being a list of pairs of integers for each token index,
            containing the start and end character index in the original context.
        context:
            The context of the example.
        max_answer_length:
            The maximum length of the answer.
        num_best_logits:
            The number of best logits to consider. Note that this function will run in
            O(`num_best_logits` ^ 2) time.
        min_null_score:
            The minimum score an answer can have.

    Returns:
        A list of the valid answers, each being a dictionary with keys "text" and
        "score", the score being the sum of the start and end logits.
    """
    # Fetch the top-k predictions for the start- and end token indices
    start_indexes = np.argsort(start_logits)[-1 : -num_best_logits - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -num_best_logits - 1 : -1].tolist()

    # We loop over all combinations of starting and ending indexes for valid answers
    valid_answers = list()
    for start_index in start_indexes:
        for end_index in end_indexes:
            # If the starting or ending index is out-of-scope, meaning that they are
            # either out of bounds or correspond to part of the input_ids that are not
            # in the context, then we skip this index
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or tuple(offset_mapping[start_index]) == (-1, -1)
                or tuple(offset_mapping[end_index]) == (-1, -1)
            ):
                continue

            # Do not consider answers with a length that is either negative or greater
            # than the context length
            max_val = max_answer_length + start_index - 1
            if end_index < start_index or end_index > max_val:
                continue

            # If we got to this point then the answer is valid, so we store the
            # corresponding start- and end character indices in the original context,
            # and from these extract the answer
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            text = context[start_char:end_char]

            # Compute the score of the answer, being the sum of the start and end
            # logits. Intuitively, this indicates how likely the answer is to be
            # correct, and allows us to pick the best valid answer.
            score = start_logits[start_index] + end_logits[end_index]

            # Add the answer to the list of valid answers, if the score is greater
            # than the minimum null score
            if score > min_null_score:
                valid_answers.append(dict(score=score, text=text))

    return valid_answers
