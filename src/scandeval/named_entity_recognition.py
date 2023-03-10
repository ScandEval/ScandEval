"""Named entity recognition benchmark dataset."""

import logging
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from numpy._typing import NDArray
from transformers import BatchEncoding
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.tokenization_utils import PreTrainedTokenizer

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark

# Set up logger
logger = logging.getLogger(__name__)


class NamedEntityRecognition(BenchmarkDataset):
    """Named entity recognition benchmark dataset.

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
        labels_in_train: Set[str] = {
            tag for tag_list in dataset_dict["train"]["ner_tags"] for tag in tag_list
        }
        self.has_misc_tags = "B-MISC" in labels_in_train or "I-MISC" in labels_in_train

        # Return the dataset dictionary
        return dataset_dict

    def _compute_metrics(
        self,
        predictions_and_labels: Tuple[NDArray, NDArray],
        id2label: Optional[Sequence[str]] = None,
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
            raw_predictions: NDArray = np.argmax(predictions, axis=-1)

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

        # Raise error if the metrics are invalid
        if results is None or results_no_misc is None:
            raise InvalidBenchmark(
                "The predictions and labels are not of the same length."
            )

        return dict(
            micro_f1=results["overall_f1"],
            micro_f1_no_misc=results_no_misc["overall_f1"],
        )

    def _tokenize_and_align_labels(
        self, examples: dict, tokenizer: PreTrainedTokenizer, label2id: Dict[str, int]
    ) -> BatchEncoding:
        """Tokenise all texts and align the labels with them.

        Args:
            examples (dict):
                The examples to be tokenised.
            tokenizer (Hugging Face tokenizer):
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts NER tags to IDs.

        Returns:
            BatchEncoding:
                A dictionary containing the tokenized data as well as labels.
        """
        # Tokenize the texts. We use the `is_split_into_words` argument here because
        # the texts in our dataset are lists of words (with a label for each word)
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=True,
        )

        # Extract a mapping between all the tokens and their corresponding word. If the
        # tokenizer is of a "fast" variant then this can be accessed through the
        # `word_ids` method. Otherwise, we have to extract it manually.
        all_labels: List[List[int]] = list()
        labels: List[str]
        word_ids: List[Optional[int]]
        for i, labels in enumerate(examples["ner_tags"]):
            # Try to get the word IDs from the tokenizer
            try:
                word_ids = tokenized_inputs.word_ids(batch_index=i)

            # If the tokenizer is not of a "fast" variant, we have to extract the word
            # IDs manually
            except ValueError:
                # Get the list of words in the document
                words: List[str] = examples["tokens"][i]

                # Get the list of token IDs in the document
                tok_ids: List[int] = tokenized_inputs.input_ids[i]

                # Decode the token IDs
                tokens = tokenizer.convert_ids_to_tokens(tok_ids)

                # Remove prefixes from the tokens
                prefixes_to_remove = ["▁", "##"]
                for tok_idx, tok in enumerate(tokens):
                    if tok:
                        for prefix in prefixes_to_remove:
                            if tok.startswith(prefix):
                                tokens[tok_idx] = tok[len(prefix) :]

                # Replace UNK tokens with the correct word
                tokens = self._handle_unk_tokens(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    words=words,
                )

                # Get list of special tokens. Some tokenizers do not record these
                # properly, which is why we convert the values to their indices and
                # then back to strings
                sp_toks = [
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.convert_tokens_to_ids(sp_tok)
                    )
                    for sp_tok in tokenizer.special_tokens_map.values()
                ]

                # Replace special tokens with `None`
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

            previous_word_idx: Optional[int] = None
            label_ids: List[int] = list()
            for word_id in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function
                if word_id is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_id != previous_word_idx:
                    label = labels[word_id]
                    try:
                        label_id = label2id[label.upper()]
                    except KeyError:
                        msg = f"The label {label} was not found in the model's config."
                        raise InvalidBenchmark(msg)
                    label_ids.append(label_id)

                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)

                previous_word_idx = word_id

            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def _handle_unk_tokens(
        self, tokenizer: PreTrainedTokenizer, tokens: List[str], words: List[str]
    ) -> List[str]:
        """Replace unknown tokens in the tokens with the corresponding word.

        Args:
            tokenizer (PreTrainedTokenizer):
                The tokenizer used to tokenize the words.
            tokens (list of str):
                The list of tokens.
            words (list of str):
                The list of words.

        Returns:
            list of str:
                The list of tokens with unknown tokens replaced by the corresponding
                word.
        """
        # Locate the token indices of the unknown tokens
        token_unk_idxs = [
            i for i, tok in enumerate(tokens) if tok == tokenizer.unk_token
        ]

        # Locate the word indices of the words which contain an unknown token
        word_unk_idxs = [
            i
            for i, word in enumerate(words)
            if tokenizer.unk_token
            in tokenizer.convert_ids_to_tokens(
                tokenizer.encode(word, add_special_tokens=False)
            )
        ]

        # Iterate over the token index and word index pairs
        for tok_idx, word_idx in zip(token_unk_idxs, word_unk_idxs):
            # Fetch the word
            word = words[word_idx]

            # Tokenize the word, which is now a list containing at least one UNK token
            tokens_with_unk = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(word, add_special_tokens=False)
            )

            # Iterate over the tokens in the word
            for possible_unk_token in tokens_with_unk:
                # If the token is not an UNK token then we remove the first occurence
                # of the content of this token from the word. The result of the `word`
                # variable will be the content of the UNK token.
                # NOTE: This is a bit hacky and not bulletproof. For instance, if the
                # word is "1925-1950" and the tokenizer splits it into ["[UNK]", "-",
                # "19", "50"], then the result will be 2519 instead of 1925. This
                # happens almost never, however, so we can live with it.
                if possible_unk_token != tokenizer.unk_token:
                    word = word.replace(possible_unk_token, "", 1)

            # Replace the token with the word
            tokens[tok_idx] = word

        # Return the tokens
        return tokens

    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
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
        map_fn = partial(
            self._tokenize_and_align_labels,
            tokenizer=kwargs["tokenizer"],
            label2id=kwargs["config"].label2id,
        )
        tokenised_dataset: Dataset = dataset.map(
            map_fn, batched=True, load_from_cache_file=False
        )
        return tokenised_dataset

    def _load_data_collator(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (PreTrainedTokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
