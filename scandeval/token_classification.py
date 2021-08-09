'''Abstract token classification benchmark'''

from transformers import (DataCollatorForTokenClassification,
                          PreTrainedTokenizerBase)
from datasets import Dataset, load_metric
from functools import partial
from typing import Optional
import logging
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

from .base import BaseBenchmark
from .utils import InvalidBenchmark


logger = logging.getLogger(__name__)


class TokenClassificationBenchmark(BaseBenchmark, ABC):
    '''Abstract token classification benchmark.

    Args:
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        learning_rate (float, optional):
            What learning rate to use when finetuning the models. Defaults to
            2e-5.
        warmup_steps (int, optional):
            The number of training steps in which the learning rate will be
            warmed up, meaning starting from nearly 0 and progressing up to
            `learning_rate` after `warmup_steps` many steps. Defaults to 50.
        batch_size (int, optional):
            The batch size used while finetuning. Defaults to 16.
        verbose (bool, optional):
            Whether to print additional output during evaluation. Defaults to
            False.

    Attributes:
        cache_dir (str): Directory where models are cached.
        learning_rate (float): Learning rate used while finetuning.
        warmup_steps (int): Number of steps used to warm up the learning rate.
        batch_size (int): The batch size used while finetuning.
        epochs (int): The number of epochs to finetune.
        num_labels (int): The number of NER labels in the dataset.
        label2id (dict): Conversion dict from NER labels to their indices.
        id2label (dict): Conversion dict from NER label indices to the labels.
    '''
    def __init__(self,
                 id2label: list,
                 epochs: int,
                 cache_dir: str = '.benchmark_models',
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 50,
                 batch_size: int = 16,
                 verbose: bool = False):
        self._metric = load_metric('seqeval')
        super().__init__(task='token-classification',
                         num_labels=len(id2label),
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         epochs=epochs,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
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
                    label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label = labels[word_idx]
                    try:
                        label_id = label2id[label]
                    except KeyError:
                        err_msg = (f'The label {label} was not found in '
                                   f'the model\'s config.')
                        raise InvalidBenchmark(err_msg)
                    label_ids.append(label_id)

                # For the other tokens in a word, we set the label to -100
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
        return DataCollatorForTokenClassification(tokenizer)

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

        processed = model.pipe(itr,
                               batch_size=self.batch_size)
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
