'''Abstract token classification benchmark'''

from transformers import (DataCollatorForTokenClassification,
                          PreTrainedTokenizerBase)
from datasets import Dataset, load_metric
from functools import partial
from typing import Optional, Dict, List
import logging
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

from .base import BaseBenchmark
from ...utils import InvalidBenchmark


logger = logging.getLogger(__name__)


class TokenClassificationBenchmark(BaseBenchmark, ABC):
    '''Abstract token classification benchmark.

    Args:
        name (str):
            The name of the dataset.
        metric_names (dict):
            A dictionary with the variable names of the metrics used in the
            dataset as keys, and a more human readable name of them as values.
        id2label (list or None, optional):
            A list of all the labels, which is used to convert indices to their
            labels. This will only be used if the pretrained model does not
            already have one. Defaults to None.
        label_synonyms (list of lists of str or None, optional):
            A list of synonyms for each label. Every entry in `label_synonyms`
            is a list of synonyms, where one of the synonyms is contained in
            `id2label`. If None then no synonyms will be used. Defaults to
            None.
        evaluate_train (bool, optional):
            Whether the models should be evaluated on the training scores.
            Defaults to False.
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        two_labels (bool, optional):
            Whether two labels should be predicted in the dataset.  If this is
            True then `split_point` has to be set. Defaults to False.
        split_point (int or None, optional):
            When there are two labels to be predicted, this is the index such
            that `id2label[:split_point]` contains the labels for the first
            label, and `id2label[split_point]` contains the labels for the
            second label. Only relevant if `two_labels` is True. Defaults to
            None.
        verbose (bool, optional):
            Whether to print additional output during evaluation. Defaults to
            False.

    Attributes:
        name (str): The name of the dataset.
        task (str): The type of task to be benchmarked.
        metric_names (dict): The names of the metrics.
        id2label (dict or None): A dictionary converting indices to labels.
        label2id (dict or None): A dictionary converting labels to indices.
        num_labels (int or None): The number of labels in the dataset.
        label_synonyms (list of lists of str): Synonyms of the dataset labels.
        evaluate_train (bool): Whether the training set should be evaluated.
        cache_dir (str): Directory where models are cached.
        two_labels (bool): Whether two labels should be predicted.
        split_point (int or None): Splitting point of `id2label` into labels.
        verbose (bool): Whether to print additional output.
    '''
    def __init__(self,
                 name: str,
                 metric_names: Dict[str, str],
                 id2label: list,
                 label_synonyms: Optional[List[List[str]]] = None,
                 evaluate_train: bool = False,
                 cache_dir: str = '.benchmark_models',
                 two_labels: bool = False,
                 split_point: Optional[int] = None,
                 verbose: bool = False):
        self._metric = load_metric('seqeval')
        super().__init__(task='token-classification',
                         name=name,
                         metric_names=metric_names,
                         id2label=id2label,
                         label_synonyms=label_synonyms,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         two_labels=two_labels,
                         split_point=split_point,
                         verbose=verbose)

    def _tokenize_and_align_labels(self,
                                   examples: dict,
                                   tokenizer,
                                   label2id: dict):
        '''Tokenise all texts and align the labels with them.

        Args:
            examples (dict):
                The examples to be tokenised.
            tokenizer (HuggingFace tokenizer):
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts NER tags to IDs.

        Returns:
            dict:
                A dictionary containing the tokenized data as well as labels.
        '''
        tokenized_inputs = tokenizer(
            examples['tokens'],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word)
            is_split_into_words=True,
            truncation=True,
            padding=True
        )
        all_labels = []
        for i, labels in enumerate(examples['orig_labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:

                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss
                # function
                if word_idx is None:
                    if self.two_labels:
                        label_ids.append([-100, -100])
                    else:
                        label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label = labels[word_idx]
                    if self.two_labels:
                        try:
                            label_id1 = label2id[label[0]]
                        except KeyError:
                            msg = (f'The label {label[0]} was not found '
                                   f'in the model\'s config.')
                            raise InvalidBenchmark(msg)
                        try:
                            label_id2 = label2id[label[1]]
                        except KeyError:
                            msg = (f'The label {label[1]} was not found '
                                   f'in the model\'s config.')
                            raise InvalidBenchmark(msg)
                        label_id = [label_id1, label_id2]

                    else:
                        try:
                            label_id = label2id[label]
                        except KeyError:
                            msg = (f'The label {label} was not found '
                                   f'in the model\'s config.')
                            raise InvalidBenchmark(msg)
                    label_ids.append(label_id)

                # For the other tokens in a word, we set the label to -100
                else:
                    if self.two_labels:
                        label_ids.append([-100, -100])
                    else:
                        label_ids.append(-100)

                previous_word_idx = word_idx

            all_labels.append(label_ids)
        tokenized_inputs['labels'] = all_labels
        return tokenized_inputs

    def _preprocess_data(self,
                         dataset: Dataset,
                         framework: str,
                         **kwargs) -> Dataset:
        '''Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset (HuggingFace dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in
                preprocessing the dataset.

        Returns:
            HuggingFace dataset: The preprocessed dataset.
        '''
        if framework in ['pytorch', 'tensorflow', 'jax']:
            map_fn = partial(self._tokenize_and_align_labels,
                             tokenizer=kwargs['tokenizer'],
                             label2id=kwargs['config'].label2id)
            tokenised_dataset = dataset.map(map_fn, batched=True)
            return tokenised_dataset
        elif framework == 'spacy':
            return dataset

    def _load_data_collator(
            self,
            tokenizer: Optional[PreTrainedTokenizerBase] = None):
        '''Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (HuggingFace tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not
                used in the initialisation of the data collator. Defaults to
                None.

        Returns:
            HuggingFace data collator: The data collator.
        '''
        if self.two_labels:
            params = dict(label_pad_token_id=[-100, -100])
        else:
            params = dict(label_pad_token_id=-100)
        return DataCollatorForTokenClassification(tokenizer, **params)

    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        '''Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model): The model.
            dataset (HuggingFace dataset): The dataset.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the
                second array contains the true labels.
        '''
        # Initialise progress bar
        if progress_bar:
            itr = tqdm(dataset['doc'])
        else:
            itr = dataset['doc']

        processed = model.pipe(itr, batch_size=32)
        map_fn = self._extract_spacy_predictions
        predictions = map(map_fn, zip(dataset['tokens'], processed))

        return list(predictions), dataset['orig_labels']

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        '''Helper function that extracts the predictions from a SpaCy model.

        Aside from extracting the predictions from the model, it also aligns
        the predictions with the gold tokens, in case the SpaCy tokeniser
        tokenises the text different from those.

        Args:
            tokens_processed (tuple):
                A pair of the labels, being a list of strings, and the SpaCy
                processed document, being a Spacy `Doc` instance.

        Returns:
            list:
                A list of predictions for each token, of the same length as the
                gold tokens (first entry of `tokens_processed`).
        '''
        tokens, processed = tokens_processed

        # Get the token labels
        token_labels = self._get_spacy_token_labels(processed)

        # Get the alignment between the SpaCy model's tokens and the gold
        # tokens
        token_idxs = [tok_idx for tok_idx, tok in enumerate(tokens)
                      for _ in str(tok)]
        pred_token_idxs = [tok_idx for tok_idx, tok in enumerate(processed)
                           for _ in str(tok)]
        alignment = list(zip(token_idxs, pred_token_idxs))

        # Get the aligned predictions
        predictions = list()
        for tok_idx, _ in enumerate(tokens):
            aligned_pred_token = [pred_token_idx
                                  for token_idx, pred_token_idx in alignment
                                  if token_idx == tok_idx][0]
            predictions.append(token_labels[aligned_pred_token])

        return predictions

    @abstractmethod
    def _get_spacy_token_labels(self, processed) -> list:
        '''Function that extracts the desired predictions from a SpaCy Doc.

        Args:
            processed (SpaCy Doc instance):
                The processed text, from the output of a SpaCy model applied on
                some text.

        Returns:
            list:
                A list of labels, for each SpaCy token.
        '''
        pass
