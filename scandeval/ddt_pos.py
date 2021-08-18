'''POS evaluation of a language model on the DDT dataset'''

from datasets import Dataset
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

from .token_classification import TokenClassificationBenchmark
from .datasets import load_ddt_pos
from .utils import doc_inherit


logger = logging.getLogger(__name__)


class DdtPosBenchmark(TokenClassificationBenchmark):
    '''Benchmark of language models on the POS part of the DDT dataset.

    Args:
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        evaluate_train (bool, optional):
            Whether the models should be evaluated on the training scores.
            Defaults to False.
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
                 cache_dir: str = '.benchmark_models',
                 evaluate_train: bool = False,
                 verbose: bool = False):
        id2label = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP',
                    'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                    'PUNCT', 'SYM', 'X']
        super().__init__(name='the POS part of DDT',
                         metric_names=dict(accuracy='Accuracy'),
                         id2label=id2label,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_ddt_pos()
        train_dict = dict(doc=X_train['doc'],
                          tokens=X_train['tokens'],
                          orig_labels=y_train['pos_tags'])
        test_dict = dict(doc=X_test['doc'],
                         tokens=X_test['tokens'],
                         orig_labels=y_test['pos_tags'])
        train = Dataset.from_dict(train_dict)
        test = Dataset.from_dict(test_dict)
        return train, test

    @doc_inherit
    def _compute_metrics(self,
                         predictions_and_labels: tuple,
                         id2label: Optional[dict] = None) -> Dict[str, float]:
        # Get the predictions from the model
        predictions, labels = predictions_and_labels

        if id2label is not None:
            raw_predictions = np.argmax(predictions, axis=-1)

            # Remove ignored index (special tokens)
            predictions = [
                [id2label[pred] for pred, lbl in zip(prediction, label)
                 if lbl != -100]
                for prediction, label in zip(raw_predictions, labels)
            ]
            labels = [
                [id2label[lbl] for _, lbl in zip(prediction, label)
                 if lbl != -100]
                for prediction, label in zip(raw_predictions, labels)
            ]

        results = self._metric.compute(predictions=predictions,
                                       references=labels)
        return dict(accuracy=results['overall_accuracy'])

    @doc_inherit
    def _get_spacy_token_labels(self, processed) -> List[str]:
        return [tok.pos_ for tok in processed]
