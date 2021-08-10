'''Sentiment evaluation of a language model on the Europarl2 dataset'''

from datasets import Dataset
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

from .text_classification import TextClassificationBenchmark
from .datasets import load_europarl2
from .utils import doc_inherit, InvalidBenchmark


logger = logging.getLogger(__name__)


class Europarl2Benchmark(TextClassificationBenchmark):
    '''Benchmark of language models on the Europarl2 dataset.

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
                 cache_dir: str = '.benchmark_models',
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 50,
                 batch_size: int = 16,
                 verbose: bool = False):
        id2label = ['neutral', 'positiv', 'negativ']
        super().__init__(epochs=5,
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_europarl2()
        train_dict = dict(doc=X_train['text'],
                          orig_label=y_train['label'])
        test_dict = dict(doc=X_test['text'],
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
                                       references=labels,
                                       average='macro')
        return dict(macro_f1=results['f1'])

    @doc_inherit
    def _log_metrics(self,
                     metrics: Dict[str, List[Dict[str, float]]],
                     model_id: str):
        kwargs = dict(metrics=metrics, metric_name='macro_f1')
        train_mean, train_std_err = self._get_stats(split='train', **kwargs)
        test_mean, test_std_err = self._get_stats(split='test', **kwargs)

        # Multiply scores by x100 to make them easier to read
        train_mean *= 100
        test_mean *= 100
        train_std_err *= 100
        test_std_err *= 100

        if not np.isnan(train_std_err):
            msg = (f'Mean macro-average F1-scores on Europarl2 for '
                   f'{model_id}:\n'
                   f'  - Train: {train_mean:.2f} ± {train_std_err:.2f}\n'
                   f'  - Test: {test_mean:.2f} ± {test_std_err:.2f}')
        else:
            msg = (f'Macro-average F1-scores on Europarl2 for {model_id}:\n'
                   f'  - Train: {train_mean:.2f}\n'
                   f'  - Test: {test_mean:.2f}')

        logger.info(msg)

    @doc_inherit
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        raise InvalidBenchmark('Evaluation of sentiment predictions '
                               'for SpaCy models is not yet implemented.')
