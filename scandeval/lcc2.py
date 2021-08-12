'''Sentiment evaluation of a language model on the LCC2 dataset'''

from datasets import Dataset
from typing import Tuple, Dict, List, Optional
import logging

from .text_classification import TextClassificationBenchmark
from .datasets import load_lcc2
from .utils import doc_inherit, InvalidBenchmark


logger = logging.getLogger(__name__)


class Lcc2Benchmark(TextClassificationBenchmark):
    '''Benchmark of language models on the LCC2 dataset.

    Args:
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        learning_rate (float, optional):
            What learning rate to use when finetuning the models. Defaults to
            2e-5.
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
                 batch_size: int = 16,
                 evaluate_train: bool = False,
                 verbose: bool = False):
        id2label = ['neutral', 'positiv', 'negativ']
        super().__init__(epochs=50,
                         warmup_steps=0,
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_lcc2()
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
        scores = self._get_stats(metrics, 'macro_f1')
        test_score, test_se = scores['test']
        test_score *= 100
        test_se *= 100

        msg = (f'Macro-average F1-scores on LCC2 for {model_id}:\n'
               f'  - Test: {test_score:.2f} ± {test_se:.2f}')

        if 'train' in scores.keys():
            train_score, train_se = scores['train']
            train_score *= 100
            train_se *= 100
            msg += f'\n  - Train: {train_score:.2f} ± {train_se:.2f}'

        logger.info(msg)

    @doc_inherit
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        raise InvalidBenchmark('Evaluation of sentiment predictions '
                               'for SpaCy models is not yet implemented.')
