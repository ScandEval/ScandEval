'''Sentiment evaluation of a language model on the Europarl1 dataset'''

from datasets import Dataset
from typing import Tuple, Dict, Optional
import logging

from .text_classification import TextClassificationBenchmark
from .datasets import load_europarl1
from .utils import doc_inherit, InvalidBenchmark


logger = logging.getLogger(__name__)


class Europarl1Benchmark(TextClassificationBenchmark):
    '''Benchmark of language models on the Europarl1 dataset.

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
        id2label = ['negativ', 'neutral', 'positiv']
        super().__init__(name='Europarl1',
                         metric_names=dict(macro_f1='Macro-average F1-score'),
                         epochs=30,
                         warmup_steps=1,
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_europarl1()
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
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        raise InvalidBenchmark('Evaluation of sentiment predictions '
                               'for SpaCy models is not yet implemented.')
