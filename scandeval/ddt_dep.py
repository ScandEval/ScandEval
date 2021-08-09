'''Dependency parsing evaluation of a language model on the DDT dataset'''

from datasets import Dataset
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
import itertools as it

from .token_classification import TokenClassificationBenchmark
from .datasets import load_ddt_dep
from .utils import doc_inherit


logger = logging.getLogger(__name__)


class DdtDepBenchmark(TokenClassificationBenchmark):
    '''Benchmark of language models on the dependency parsing part of the DDT.

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
        num_labels (int): The number of POS labels in the dataset.
        label2id (dict): Conversion dict from POS labels to their indices.
        id2label (dict): Conversion dict from POS label indices to the labels.
    '''
    def __init__(self,
                 cache_dir: str = '.benchmark_models',
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 50,
                 batch_size: int = 16,
                 verbose: bool = False):
        id2label_head = [str(i) for i in range(100)]
        id2label_dep = ['acl',
                        'acl:relcl',
                        'advcl',
                        'advmod',
                        'advmod:emph',
                        'advmod:lmod',
                        'amod',
                        'appos',
                        'aux',
                        'aux:pass',
                        'case',
                        'cc',
                        'cc:preconj',
                        'ccomp',
                        'clf',
                        'compound',
                        'compound:lvc',
                        'compound:prt',
                        'compound:redup',
                        'compound:svc',
                        'conj',
                        'cop',
                        'csubj',
                        'csubj:pass',
                        'dep',
                        'det',
                        'det:numgov',
                        'det:nummod',
                        'det:poss',
                        'discourse',
                        'dislocated',
                        'expl',
                        'expl:impers',
                        'expl:pass',
                        'expl:pv',
                        'fixed',
                        'flat',
                        'flat:foreign',
                        'flat:name',
                        'goeswith',
                        'iobj',
                        'list',
                        'mark',
                        'nmod',
                        'nmod:poss',
                        'nmod:tmod',
                        'nsubj',
                        'nsubj:pass',
                        'nummod',
                        'nummod:gov',
                        'obj',
                        'obl',
                        'obl:agent',
                        'obl:arg',
                        'obl:lmod',
                        'obl:tmod',
                        'orphan',
                        'parataxis',
                        'punct',
                        'reparandum',
                        'root',
                        'vocative',
                        'xcomp']
        id2label = list(map(str, it.product(id2label_head, id2label_dep)))
        super().__init__(epochs=5,
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_ddt_dep()
        train_labels = [list(map(str, zip(heads, deps)))
                        for heads, deps in zip(y_train['heads'],
                                               y_train['deps'])]
        test_labels = [list(map(str, zip(heads, deps)))
                       for heads, deps in zip(y_test['heads'],
                                              y_test['deps'])]
        train_dict = dict(doc=X_train['doc'],
                          tokens=X_train['tokens'],
                          orig_labels=train_labels)
        test_dict = dict(doc=X_test['doc'],
                         tokens=X_test['tokens'],
                         orig_labels=test_labels)
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
    def _log_metrics(self,
                     metrics: Dict[str, List[Dict[str, float]]],
                     model_id: str):
        kwargs = dict(metrics=metrics, metric_name='accuracy')
        train_mean, train_std_err = self._get_stats(split='train', **kwargs)
        test_mean, test_std_err = self._get_stats(split='test', **kwargs)

        # Multiply scores by x100 to make them easier to read
        train_mean *= 100
        test_mean *= 100
        train_std_err *= 100
        test_std_err *= 100

        if not np.isnan(train_std_err):
            msg = (f'Mean LAS on the DEP part of DDT for {model_id}:\n'
                   f'  - Train: {train_mean:.2f} ± {train_std_err:.2f}\n'
                   f'  - Test: {test_mean:.2f} ± {test_std_err:.2f}')
        else:
            msg = (f'LAS on the DEP part of DDT for {model_id}:\n'
                   f'  - Train: {train_mean:.2f}\n'
                   f'  - Test: {test_mean:.2f}')

        logger.info(msg)

    @doc_inherit
    def _get_spacy_token_labels(self, processed) -> List[str]:
        def get_heads_and_deps(token) -> Tuple[str, str]:
            dep = token.dep_.lower().replace('obl:loc', 'obl:lmod')
            if dep == 'root':
                head = '0'
            else:
                head = str(token.head.i + 1)
            return (head, dep)
        return [str(get_heads_and_deps(token)) for token in processed]
