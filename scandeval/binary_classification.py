'''Abstract binary classification benchmark'''

from datasets import Dataset
import logging
from abc import ABC
from typing import Optional, Dict, List

from .text_classification import TextClassificationBenchmark
from .utils import InvalidBenchmark


logger = logging.getLogger(__name__)


class BinaryClassificationBenchmark(TextClassificationBenchmark, ABC):
    '''Abstract binary classification benchmark.

    Args:
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
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
                 id2label: List[str],
                 cache_dir: str = '.benchmark_models',
                 evaluate_train: bool = False,
                 verbose: bool = False):
        label_synonyms = [
            ['LABEL_0', 'negative', 'negativ', 'falsk', 'false', 'objective',
                'hlutlægt', id2label[0]],
            ['LABEL_1', 'positive', 'positiv', 'sand', 'true', 'subjective',
                'huglægt', id2label[1]]
        ]
        super().__init__(name=name,
                         metric_names=dict(f1='F1-score'),
                         id2label=id2label,
                         label_synonyms=label_synonyms,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         two_labels=False,
                         verbose=verbose)

    def _compute_metrics(self,
                         predictions_and_labels: tuple,
                         id2label: Optional[list] = None) -> Dict[str, float]:
        '''Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions and the
                second array contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the
                metric values as values.
        '''
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)
        results = self._metric.compute(predictions=predictions,
                                       references=labels)
        return dict(f1=results['f1'])

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
        raise InvalidBenchmark('Evaluation of text classification predictions '
                               'for SpaCy models is not yet implemented.')
