"""Named entity recognition benchmark dataset."""

import itertools as it
import json
import logging
import random
import re
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedModel
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.modeling_utils import ModelOutput

from .benchmark_dataset import BenchmarkDataset
from .exceptions import InvalidBenchmark, NeedsExtraInstalled
from .generation import extract_raw_predictions
from .protocols import GenerativeModel, Tokenizer
from .types import Labels, Predictions
from .utils import (
    GENERATIVE_MODEL_TASKS,
    model_is_generative,
    raise_if_model_output_contains_nan_values,
)

try:
    import demjson3
except ImportError:
    demjson3 = None

logger = logging.getLogger(__package__)


class NamedEntityRecognition(BenchmarkDataset):
    """Named entity recognition benchmark dataset.

    Args:
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Attributes:
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
    """

    def _process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Process the data.

        Args:
            dataset_dict:
                The dataset dictionary.

        Returns:
            The processed dataset dictionary.
        """
        # Check what labels are present in the dataset, and store if MISC tags are not
        # present
        labels_in_train: set[str] = {
            tag for tag_list in dataset_dict["train"]["labels"] for tag in tag_list
        }
        self.has_misc_tags = "B-MISC" in labels_in_train or "I-MISC" in labels_in_train

        return dataset_dict

    def _compute_metrics(
        self, model_outputs_and_labels: tuple[Predictions, Labels], id2label: list[str]
    ) -> dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            model_outputs_and_labels:
                The first array contains the probability predictions and the second
                array contains the true labels.
            id2label:
                Conversion of indices to labels.

        Returns:
            A dictionary with the names of the metrics as keys and the metric values as
            values.
        """
        model_outputs, labels = model_outputs_and_labels

        raise_if_model_output_contains_nan_values(model_output=model_outputs)

        predictions: list[list[str]]
        if not isinstance(model_outputs[0][0], str):
            raw_predictions: list[list[int]] = np.argmax(
                model_outputs, axis=-1
            ).tolist()

            # Remove ignored index (special tokens)
            predictions = [
                [
                    id2label[pred_id]
                    for pred_id, lbl_id in zip(  # type: ignore[call-overload]
                        pred, label
                    )
                    if lbl_id != -100
                ]
                for pred, label in zip(raw_predictions, labels)
            ]
            labels = [
                [
                    (
                        id2label[lbl_id]
                        if isinstance(lbl_id, int) or isinstance(lbl_id, np.int_)
                        else lbl_id
                    )
                    for lbl_id in label  # type: ignore[call-overload]
                    if lbl_id != -100
                ]
                for label in labels
            ]

        else:
            predictions = model_outputs  # type: ignore[assignment]

        # Replace predicted tag with either MISC or O tags if they are not part of the
        # dataset
        id2label_without_misc = set(self.dataset_config.id2label).difference(
            {"b-misc", "i-misc"}
        )
        ner_tag: str
        for i, prediction_list in enumerate(predictions):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag not in id2label_without_misc:
                    if self.has_misc_tags and ner_tag[:2] == "b-":
                        predictions[i][j] = "b-misc"
                    elif self.has_misc_tags and ner_tag[:2] == "i-":
                        predictions[i][j] = "i-misc"
                    else:
                        predictions[i][j] = "o"

        # Remove MISC labels from predictions
        predictions_no_misc = deepcopy(predictions)
        for i, prediction_list in enumerate(predictions_no_misc):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag[-4:] == "misc":
                    predictions_no_misc[i][j] = "o"

        # Remove MISC labels from labels
        labels_no_misc = deepcopy(labels)
        for i, label_list in enumerate(labels_no_misc):
            for j, ner_tag in enumerate(label_list):  # type: ignore[arg-type]
                if (
                    isinstance(ner_tag, str)
                    and len(ner_tag) >= 4
                    and ner_tag[-4:] == "misc"  # type: ignore[index]
                ):
                    labels_no_misc[i][j] = "o"  # type: ignore[call-overload]

        # Compute the metrics
        # We manually set the F1 metric to be 100% if both the labels and the models
        # have no NER tags in them, since this causes an error with the `compute`
        # method otherwise
        predictions_all_zero = all(
            all(ner_tag == "o" for ner_tag in prediction_list)
            for prediction_list in predictions
        )
        labels_all_zero = all(
            all(ner_tag == "o" for ner_tag in label_list) for label_list in labels
        )
        if predictions_all_zero and labels_all_zero:
            results = dict(overall_f1=1.0)
        else:
            results = self._metrics["micro_f1"].compute(
                predictions=predictions, references=labels
            )

        # Compute the metrics without MISC tags
        # We manually set the F1 metric to be 100% if both the labels and the models
        # have no NER tags in them, since this causes an error with the `compute`
        # method otherwise
        predictions_no_misc_all_zero = all(
            all(ner_tag == "o" for ner_tag in prediction_list)
            for prediction_list in predictions_no_misc
        )
        labels_no_misc_all_zero = all(
            all(ner_tag == "o" for ner_tag in label_list)
            for label_list in labels_no_misc
        )
        if predictions_no_misc_all_zero and labels_no_misc_all_zero:
            results_no_misc = dict(overall_f1=1.0)
        else:
            results_no_misc = self._metrics["micro_f1_no_misc"].compute(
                predictions=predictions_no_misc, references=labels_no_misc
            )

        # Raise error if the metrics are invalid
        if results is None or results_no_misc is None:
            raise InvalidBenchmark(
                "The predictions and labels are not of the same length."
            )

        return dict(
            micro_f1_no_misc=results_no_misc["overall_f1"],
            micro_f1=results["overall_f1"],
        )

    def _tokenize_and_align_labels(
        self, examples: dict, tokenizer: Tokenizer, label2id: dict[str, int]
    ) -> BatchEncoding:
        """Tokenise all texts and align the labels with them.

        Args:
            examples:
                The examples to be tokenised.
            tokenizer:
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts NER tags to IDs.

        Returns:
            A dictionary containing the tokenized data as well as labels.
        """
        # Tokenize the texts. We use the `is_split_into_words` argument here because
        # the texts in our dataset are lists of words (with a label for each word)
        tokenized_inputs = tokenizer(
            examples["tokens"], is_split_into_words=True, truncation=True, padding=True
        )

        # Extract a mapping between all the tokens and their corresponding word. If the
        # tokenizer is of a "fast" variant then this can be accessed through the
        # `word_ids` method. Otherwise, we have to extract it manually.
        all_labels: list[list[int]] = list()
        labels: list[str]
        word_ids: list[int | None]
        for i, labels in enumerate(examples["labels"]):
            # Try to get the word IDs from the tokenizer
            try:
                word_ids = tokenized_inputs.word_ids(batch_index=i)

            # If the tokenizer is not of a "fast" variant, we have to extract the word
            # IDs manually
            except ValueError:
                # Get the list of words in the document
                words: list[str] = examples["tokens"][i]

                # Get the list of token IDs in the document
                tok_ids: list[int] = tokenized_inputs.input_ids[i]

                # Decode the token IDs
                tokens = tokenizer.convert_ids_to_tokens(tok_ids)
                assert isinstance(tokens, list)

                # Remove prefixes from the tokens
                prefixes_to_remove = ["▁", "##"]
                for tok_idx, tok in enumerate(tokens):
                    if tok:
                        for prefix in prefixes_to_remove:
                            if tok.startswith(prefix):
                                tokens[tok_idx] = tok[len(prefix) :]

                # Replace UNK tokens with the correct word
                tokens = self._handle_unk_tokens(
                    tokenizer=tokenizer, tokens=tokens, words=words
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
                tokens_with_none = [None if tok in sp_toks else tok for tok in tokens]

                # Get the alignment between the words and the tokens, on a character
                # level
                word_idxs = [
                    word_idx for word_idx, word in enumerate(words) for _ in str(word)
                ]
                token_idxs = [
                    tok_idx
                    for tok_idx, tok_or_none in enumerate(tokens_with_none)
                    for _ in str(tok_or_none)
                    if tok_or_none is not None
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
                for tok_idx, tok_or_none in enumerate(tokens_with_none):
                    if tok_or_none is None or tok_or_none == "":
                        word_ids.append(None)
                    else:
                        word_idx = [
                            word_idx
                            for word_idx, token_idx in alignment
                            if token_idx == tok_idx
                        ][0]
                        word_ids.append(word_idx)

            previous_word_idx: int | None = None
            label_ids: list[int] = list()
            for word_id in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function
                if word_id is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_id != previous_word_idx:
                    label = labels[word_id]
                    try:
                        label_id = label2id[label.lower()]
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
        self, tokenizer: Tokenizer, tokens: list[str], words: list[str]
    ) -> list[str]:
        """Replace unknown tokens in the tokens with the corresponding word.

        Args:
            tokenizer:
                The tokenizer used to tokenize the words.
            tokens:
                The list of tokens.
            words:
                The list of words.

        Returns:
            The list of tokens with unknown tokens replaced by the corresponding word.
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
            dataset:
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face dataset:
                The preprocessed dataset.
        """
        if kwargs["model_config"].task in GENERATIVE_MODEL_TASKS:
            if "few_shot_examples" in kwargs:
                few_shot_examples = kwargs["few_shot_examples"]
                few_shot_fn = partial(
                    self._apply_few_shot_prompt, few_shot_examples=few_shot_examples
                )
                dataset = dataset.map(
                    few_shot_fn,
                    batched=True,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )

            def tokenise(examples: dict) -> BatchEncoding:
                return kwargs["tokenizer"](
                    text=examples["text"], truncation=True, padding=False
                )

            tokenised_dataset = dataset.map(
                tokenise, batched=True, load_from_cache_file=False, keep_in_memory=True
            )

        else:
            map_fn = partial(
                self._tokenize_and_align_labels,
                tokenizer=kwargs["tokenizer"],
                label2id=kwargs["hf_model_config"].label2id,
            )
            tokenised_dataset = dataset.map(
                map_fn, batched=True, load_from_cache_file=False, keep_in_memory=True
            )

        return tokenised_dataset

    def _load_data_collator(
        self,
        tokenizer: Tokenizer | None = None,
        model: PreTrainedModel | GenerativeModel | None = None,
    ):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer:
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.
            model:
                A pretrained model. Can be None if the model is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        if model_is_generative(model=model):
            return DataCollatorWithPadding(tokenizer=tokenizer)
        else:
            return DataCollatorForTokenClassification(
                tokenizer=tokenizer, label_pad_token_id=-100
            )

    def _extract_few_shot_examples(
        self, train_dataset: Dataset, random_seed: int
    ) -> list[dict[str, Any]]:
        """Extract few-shot examples from the training dataset.

        Args:
            train_dataset:
                The training dataset.
            random_seed:
                The random seed to use when extracting the few-shot examples.

        Returns:
            list[dict[str, Any]]:
                The few-shot examples.
        """
        shuffled_train = train_dataset.shuffle(seed=random_seed)
        num_few_shots = self.dataset_config.num_few_shot_examples
        labels = it.cycle(
            [
                label.lower()
                for label in self.dataset_config.task.labels
                if label.lower().startswith("b-")
            ]
        )
        few_shot_examples: list[dict[str, Any]] = list()

        # We pick the few-shot examples one at a time rather than all at once since
        # we're working with a bootstrapped training dataset, meaning that it will have
        # duplicates. This ensures that we don't have any duplicates in the few-shot
        # examples
        while len(few_shot_examples) < num_few_shots:
            label = next(labels)
            possible_examples = shuffled_train.filter(
                lambda x: label in [tag.lower() for tag in x["labels"]]
            )
            if len(possible_examples) == 0:
                continue
            example = possible_examples.select(range(1))[0]
            few_shot_examples.append(example)
            shuffled_train = shuffled_train.filter(
                lambda x: x["text"] != example["text"]
            )

        random.seed(random_seed)
        random.shuffle(few_shot_examples)
        return few_shot_examples

    def _apply_few_shot_prompt(
        self, examples: dict, few_shot_examples: list[dict], tokenizer: Tokenizer
    ) -> dict:
        """Apply a few-shot prompt to the examples.

        Args:
            examples:
                The examples to apply the prompt to.
            few_shot_examples:
                The examples to be included in the few-shot prompt.
            tokenizer:
                The tokenizer to use to encode the few-shot prompt.

        Returns:
            The examples with the few-shot prompt applied.
        """

        def create_label(example: dict) -> str:
            labels: dict[str, list[str]] = {
                prompt_label: list()
                for prompt_label in self.dataset_config.prompt_label_mapping.values()
            }
            for token, label in zip(example["tokens"], example["labels"]):
                label = label.lower()
                if label == "o":
                    continue
                prompt_label = self.dataset_config.prompt_label_mapping[label]
                if label.startswith("b-"):
                    labels[prompt_label].append(token)
                elif label.startswith("i-"):
                    labels[prompt_label][-1] += " " + token
            return json.dumps(labels, ensure_ascii=False)

        # Build the few-shot part of the prompt
        few_shot_prompts = [
            self.dataset_config.prompt_template.format(
                text=" ".join(example["tokens"]).replace("\n", " ").strip(),
                label=create_label(example),
            )
            for example in few_shot_examples
        ]
        prompt_prefix = ""
        if self.dataset_config.prompt_prefix:
            prompt_prefix = self.dataset_config.prompt_prefix + "\n\n"
        few_shot_prompt = prompt_prefix + "\n\n".join(few_shot_prompts)

        # Add the texts from the examples to the prompts
        new_prompts = [
            self.dataset_config.prompt_template.format(
                text=" ".join(tokens).replace("\n", " ").strip(), label=""
            )
            for tokens in examples["tokens"]
        ]
        examples["text"] = [
            few_shot_prompt + "\n\n" + new_prompt for new_prompt in new_prompts
        ]

        return examples

    def _extract_labels_from_generation(
        self,
        input_batch: dict[str, list],
        model_output: ModelOutput,
        tokenizer: Tokenizer,
    ) -> list[Any]:
        """Extract the predicted labels from the generated output.

        Args:
            input_batch:
                The input batch, where the keys are the feature names and the values
                are lists with the feature values.
            model_output:
                The raw generated output of the model.
            tokenizer:
                The tokenizer used together with the model.

        Returns:
            list:
                The predicted labels.
        """
        if demjson3 is None:
            raise NeedsExtraInstalled(extra="generative")

        raw_predictions = extract_raw_predictions(
            generated_sequences=model_output["sequences"], tokenizer=tokenizer
        )

        # Attempt to extract the JSON dictionary from the predictions
        json_regex = r"\{.+?\}"
        json_matches = [
            re.search(pattern=json_regex, string=raw_prediction, flags=re.DOTALL)
            or raw_prediction
            for raw_prediction in raw_predictions
        ]
        raw_predictions = [
            json_match.group() if isinstance(json_match, re.Match) else json_match
            for json_match in json_matches
        ]

        tokens = input_batch["tokens"]
        predicted_labels: list[list[str]] = [
            ["o"] * len(token_ids) for token_ids in tokens
        ]
        for idx, raw_prediction in enumerate(raw_predictions):
            try:
                json_output = demjson3.decode(txt=raw_prediction)
                if not isinstance(json_output, dict):
                    logger.debug(
                        "The model output is not a JSON dictionary, so cannot parse "
                        f"it. Skipping. Here is the output: {raw_prediction}"
                    )
                    continue
                elif not all(isinstance(key, str) for key in json_output.keys()):
                    logger.debug(
                        "The model output is not a JSON dictionary with string keys, "
                        "so cannot parse it. Skipping. Here is the output: "
                        f"{raw_prediction}"
                    )
                    continue
                elif not all(isinstance(value, list) for value in json_output.values()):
                    logger.debug(
                        "The model output is not a JSON dictionary with list values, "
                        "so cannot parse it. Skipping. Here is the output: "
                        f"{raw_prediction}"
                    )
                    continue
                prediction_dict: dict[str, list[str]] = json_output
            except demjson3.JSONDecodeError:
                logger.debug(
                    "The model output is not valid JSON, so cannot parse it. Skipping. "
                    f"Here is the output: {raw_prediction!r}"
                )
                continue

            prompt_label_mapping = self.dataset_config.prompt_label_mapping
            for prompt_tag_name, named_entities in prediction_dict.items():
                try:
                    tag_name = [
                        tag[2:]
                        for tag, prompt_tag in prompt_label_mapping.items()
                        if prompt_tag == prompt_tag_name
                    ][0]
                except IndexError:
                    logger.debug(
                        "The model produced an invalid prompt tag name, "
                        f"{prompt_tag_name}. Skipping."
                    )
                    continue

                named_entities = [str(named_entity) for named_entity in named_entities]
                for named_entity in named_entities:
                    for ne_idx, named_entity_word in enumerate(named_entity.split()):
                        for token_idx, token in enumerate(tokens[idx]):
                            if named_entity_word in token:
                                if ne_idx == 0:
                                    predicted_labels[idx][token_idx] = f"b-{tag_name}"
                                elif (
                                    predicted_labels[idx][token_idx] == "o"
                                    and predicted_labels[idx][token_idx - 1][2:]
                                    == tag_name
                                ):
                                    predicted_labels[idx][token_idx] = f"i-{tag_name}"
        return predicted_labels
