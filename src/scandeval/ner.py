"""NER tagging benchmark dataset."""

import logging
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Sequence

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark

logger = logging.getLogger(__name__)


class NERBenchmark(BenchmarkDataset):
    """NER tagging benchmark.

    Args:
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.

    Attributes:
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.
    """

    def _process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Process the data.

        Args:
            dataset_dict (DatasetDict):
                The dataset dictionary.

        Returns:
            DatasetDict:
                The processed dataset dictionary.
        """
        # Check what labels are present in the dataset, and store if MISC tags are not
        # present
        labels_in_train = {
            tag for tag_list in dataset_dict["train"]["ner_tags"] for tag in tag_list
        }
        self.has_misc_tags = "B-MISC" in labels_in_train or "I-MISC" in labels_in_train

        # Return the dataset dictionary
        return dataset_dict

    def _compute_metrics(
        self, predictions_and_labels: tuple, id2label: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions and the second
                array contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        # Get the predictions from the model
        predictions, labels = predictions_and_labels

        if id2label is not None:
            raw_predictions = np.argmax(predictions, axis=-1)

            # Remove ignored index (special tokens)
            predictions = [
                [
                    id2label[pred_id]
                    for pred_id, lbl_id in zip(pred, label)
                    if lbl_id != -100
                ]
                for pred, label in zip(raw_predictions, labels)
            ]
            labels = [
                [id2label[lbl_id] for _, lbl_id in zip(pred, label) if lbl_id != -100]
                for pred, label in zip(raw_predictions, labels)
            ]

        # Replace predicted tag with either MISC or O tags if they are not part of the
        # dataset
        id2label_without_misc = set(self.dataset_config.id2label).difference(
            {"B-MISC", "I-MISC"}
        )
        for i, prediction_list in enumerate(predictions):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag not in id2label_without_misc:
                    if self.has_misc_tags and ner_tag[:2] == "B-":
                        predictions[i][j] = "B-MISC"
                    elif self.has_misc_tags and ner_tag[:2] == "I-":
                        predictions[i][j] = "I-MISC"
                    else:
                        predictions[i][j] = "O"

        # Remove MISC labels from predictions
        predictions_no_misc = deepcopy(predictions)
        for i, prediction_list in enumerate(predictions_no_misc):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag[-4:] == "MISC":
                    predictions_no_misc[i][j] = "O"

        # Remove MISC labels from labels
        labels_no_misc = deepcopy(labels)
        for i, label_list in enumerate(labels_no_misc):
            for j, ner_tag in enumerate(label_list):
                if ner_tag[-4:] == "MISC":
                    labels_no_misc[i][j] = "O"

        # Compute the metrics
        results = self._metrics["micro_f1"].compute(
            predictions=predictions, references=labels
        )
        results_no_misc = self._metrics["micro_f1_no_misc"].compute(
            predictions=predictions_no_misc, references=labels_no_misc
        )

        return dict(
            micro_f1=results["overall_f1"],
            micro_f1_no_misc=results_no_misc["overall_f1"],
        )

    def _get_spacy_token_labels(self, processed) -> Sequence[str]:
        """Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.

        Returns:
            A list of strings:
                The predicted NER labels.
        """

        def get_ent(token) -> str:
            """Helper function that extracts the entity from a SpaCy token"""

            # Deal with the O tag separately, as it is the only tag not of the form
            # B-tag or I-tag
            if token.ent_iob_ == "O":
                return "O"

            # In general return a tag of the form B-tag or I-tag
            else:
                # Extract tag from spaCy token
                ent = f"{token.ent_iob_}-{token.ent_type_}"

                # Convert the tag to the its canonical synonym
                alt_idx = self.dataset_config.label2id[f"{token.ent_iob_}-MISC".upper()]
                return self.dataset_config.id2label[
                    self.dataset_config.label2id.get(ent, alt_idx)
                ]

        return [get_ent(token) for token in processed]

    def _tokenize_and_align_labels(self, examples: dict, tokenizer, label2id: dict):
        """Tokenise all texts and align the labels with them.

        Args:
            examples (dict):
                The examples to be tokenised.
            tokenizer (Hugging Face tokenizer):
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts NER tags to IDs.

        Returns:
            dict:
                A dictionary containing the tokenized data as well as labels.
        """
        tokenized_inputs = tokenizer(
            examples["tokens"],
            # We use this argument because the texts in our dataset are lists of words
            # (with a label for each word)
            is_split_into_words=True,
            truncation=True,
            padding=True,
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            try:
                word_ids = tokenized_inputs.word_ids(batch_index=i)

            # This happens if the tokenizer is not of the fast variant, in which case
            # the `word_ids` method is not available, so we have to extract this
            # manually. It's slower, but it works, and it should only occur rarely,
            # when the Hugging Face team has not implemented a fast variant of the
            # tokenizer yet.
            except ValueError:

                # Get the list of words in the document
                words = examples["tokens"][i]

                # Get the list of token IDs in the document
                tok_ids = tokenized_inputs.input_ids[i]

                # Decode the token IDs
                tokens = tokenizer.convert_ids_to_tokens(tok_ids)

                # Remove prefixes from the tokens
                prefixes_to_remove = ["▁", "##"]
                for tok_idx, tok in enumerate(tokens):
                    for prefix in prefixes_to_remove:
                        tok = tok.lstrip(prefix)
                    tokens[tok_idx] = tok

                # Replace special tokens with `None`
                sp_toks = tokenizer.special_tokens_map.values()
                tokens = [None if tok in sp_toks else tok for tok in tokens]

                # Get the alignment between the words and the tokens, on a character
                # level
                word_idxs = [
                    word_idx for word_idx, word in enumerate(words) for _ in str(word)
                ]
                token_idxs = [
                    tok_idx
                    for tok_idx, tok in enumerate(tokens)
                    for _ in str(tok)
                    if tok is not None
                ]
                alignment = list(zip(word_idxs, token_idxs))

                # Raise error if there are not as many characters in the words as in
                # the tokens. This can be due to the use of a different prefix.
                if len(word_idxs) != len(token_idxs):
                    raise InvalidBenchmark(
                        "The tokens could not be aligned with the words during manual "
                        "word-token alignment. It seems that the tokenizer is neither "
                        "of the fast variant nor of a SentencePiece/WordPiece variant."
                    )

                # Get the aligned word IDs
                word_ids = list()
                for tok_idx, tok in enumerate(tokens):
                    if tok is None or tok == "":
                        word_ids.append(None)
                    else:
                        word_idx = [
                            word_idx
                            for word_idx, token_idx in alignment
                            if token_idx == tok_idx
                        ][0]
                        word_ids.append(word_idx)

            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:

                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label = labels[word_idx]
                    try:
                        label_id = label2id[label.upper()]
                    except KeyError:
                        msg = f"The label {label} was not found in the model's config."
                        raise InvalidBenchmark(msg)
                    label_ids.append(label_id)

                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        """Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset (Hugging Face dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face dataset: The preprocessed dataset.
        """
        if framework == "pytorch":
            map_fn = partial(
                self._tokenize_and_align_labels,
                tokenizer=kwargs["tokenizer"],
                label2id=kwargs["config"].label2id,
            )
            tokenised_dataset = dataset.map(
                map_fn, batched=True, load_from_cache_file=False
            )
            return tokenised_dataset
        elif framework == "spacy":
            return dataset

    def _load_data_collator(self, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Hugging Face tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def _get_spacy_predictions_and_labels(self, model, dataset: Dataset) -> tuple:
        """Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the second
                array contains the true labels.
        """
        # Initialise progress bar
        if self.benchmark_config.progress_bar:
            itr = tqdm(dataset["doc"], desc="Evaluating model", leave=False)
        else:
            itr = dataset["doc"]

        processed = model.pipe(itr, batch_size=32)
        map_fn = self._extract_spacy_predictions
        predictions = map(map_fn, zip(dataset["tokens"], processed))

        return list(predictions), dataset["ner_tags"]

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        """Helper function that extracts the predictions from a SpaCy model.

        Aside from extracting the predictions from the model, it also aligns the
        predictions with the gold tokens, in case the SpaCy tokeniser tokenises the
        text different from those.

        Args:
            tokens_processed (tuple):
                A pair of the labels, being a list of strings, and the SpaCy processed
                document, being a Spacy `Doc` instance.

        Returns:
            list:
                A list of predictions for each token, of the same length as the gold
                tokens (first entry of `tokens_processed`).
        """
        tokens, processed = tokens_processed

        # Get the token labels
        token_labels = self._get_spacy_token_labels(processed)

        # Get the alignment between the SpaCy model's tokens and the gold tokens
        token_idxs = [tok_idx for tok_idx, tok in enumerate(tokens) for _ in str(tok)]
        pred_token_idxs = [
            tok_idx for tok_idx, tok in enumerate(processed) for _ in str(tok)
        ]
        alignment = list(zip(token_idxs, pred_token_idxs))

        # Get the aligned predictions
        predictions = list()
        for tok_idx, _ in enumerate(tokens):
            aligned_pred_token = [
                pred_token_idx
                for token_idx, pred_token_idx in alignment
                if token_idx == tok_idx
            ][0]
            predictions.append(token_labels[aligned_pred_token])

        return predictions
