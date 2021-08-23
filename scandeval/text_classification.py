'''Abstract text classification benchmark'''

from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from datasets import Dataset, load_metric
from functools import partial
import logging
from abc import ABC
from typing import Optional, Dict, List

from .base import BaseBenchmark
from .utils import InvalidBenchmark


logger = logging.getLogger(__name__)


class TextClassificationBenchmark(BaseBenchmark, ABC):
    '''Abstract text classification benchmark.

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
        self._metric = load_metric('f1')
        super().__init__(task='text-classification',
                         name=name,
                         metric_names=metric_names,
                         id2label=id2label,
                         label_synonyms=label_synonyms,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         two_labels=two_labels,
                         split_point=split_point,
                         verbose=verbose)

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
        return DataCollatorWithPadding(tokenizer, padding='longest')

    def create_numerical_labels(self, examples: dict, label2id: dict) -> dict:
        try:
            examples['label'] = [label2id[lbl]
                                 for lbl in examples['orig_label']]
        except KeyError:
            raise InvalidBenchmark(f'One of the labels in the dataset, '
                                   f'{examples["orig_label"]}, does not occur '
                                   f'in the label2id dictionary {label2id}.')
        return examples

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
            tokenizer = kwargs['tokenizer']

            def tokenise(examples: dict) -> dict:
                doc = examples['doc']
                return tokenizer(doc, truncation=True, padding=True)
            tokenised = dataset.map(tokenise, batched=True)

            numericalise = partial(self.create_numerical_labels,
                                   label2id=kwargs['config'].label2id)
            preprocessed = tokenised.map(numericalise, batched=True)

            return preprocessed.remove_columns(['doc', 'orig_label'])

        elif framework == 'spacy':
            raise InvalidBenchmark('Evaluation of text predictions '
                                   'for SpaCy models is not yet implemented.')
