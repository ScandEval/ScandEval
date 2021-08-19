'''Hate speech classification of a language model on the DKHate dataset'''

from datasets import Dataset
from typing import Tuple, Dict, Optional
import logging

from .text_classification import TextClassificationBenchmark
from .datasets import load_dkhate
from .utils import doc_inherit, InvalidBenchmark


logger = logging.getLogger(__name__)


class DkHateBenchmark(TextClassificationBenchmark):
    '''Benchmark of language models on the DKHate dataset.

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
        id2label = ['NOT', 'OFF']
        label_synonyms = [
            ['LABEL_0', 'NOT'],
            ['LABEL_1', 'OFF']
        ]
        super().__init__(name='DKHate',
                         metric_names=dict(f1='F1-score'),
                         id2label=id2label,
                         label_synonyms=label_synonyms,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_dkhate()
        train_dict = dict(doc=X_train['tweet'],
                          orig_label=y_train['label'])
        test_dict = dict(doc=X_test['tweet'],
                         orig_label=y_test['label'])
        train = Dataset.from_dict(train_dict)
        test = Dataset.from_dict(test_dict)
        return train, test

    @doc_inherit
    def _compute_metrics(self,
                         predictions_and_labels: tuple,
                         id2label: Optional[dict] = None) -> Dict[str, float]:
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)
        results = self._metric.compute(predictions=predictions,
                                       references=labels)
        return dict(f1=results['f1'])

    @doc_inherit
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        raise InvalidBenchmark('Evaluation of text classification predictions '
                               'for SpaCy models is not yet implemented.')
