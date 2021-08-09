'''Dependency parsing evaluation of a language model on the DDT dataset'''

from datasets import Dataset
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

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
        id2label = ['acl',
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
                    'xcomp'] + [str(i) for i in range(100)]
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
        train_dict = dict(doc=X_train['doc'],
                          tokens=X_train['tokens'],
                          orig_labels=list(zip(y_train['heads'],
                                               y_train['deps'])))
        test_dict = dict(doc=X_test['doc'],
                          tokens=X_test['tokens'],
                          orig_labels=list(zip(y_test['heads'],
                                               y_test['deps'])))
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
                [(id2label[pred[0]], id2label[pred[0]])
                 for pred, lbl in zip(prediction, label_list) if lbl != -100]
                for prediction, label_list in zip(raw_predictions, labels[0])
            ]
            labels = [[[id2label[lbl] for lbl in label if lbl != -100]
                      for label in labels[0]],
                     [[id2label[lbl] for lbl in label if lbl != -100]
                      for label in labels[1]]]

        # Extract the heads for UAS computation
        ref_heads = [label_lists[0] for label_lists in labels]
        pred_heads = [[head for head, _ in pred_list]
                      for pred_list in predictions]

        # Extract the heads and deps for LAS computation
        ref_deps = [label_lists[1] for label_lists in labels]
        pred_deps = [[dep for _, dep in pred_list]
                     for pred_list in predictions]
        ref_heads_deps = [list(map(str, zip(ref_h, ref_d)))
                          for ref_h , ref_d in zip(ref_heads, ref_deps)]
        pred_heads_deps = [list(map(str, zip(pred_h , pred_d)))
                          for pred_h, pred_d in zip(pred_heads, pred_deps)]

        # Compute the metrics
        uas_results = self._metric.compute(predictions=pred_heads,
                                            references=ref_heads)
        las_results = self._metric.compute(predictions=pred_heads_deps,
                                           references=ref_heads_deps)

        return dict(uas=uas_results['overall_accuracy'],
                    las=las_results['overall_accuracy'])

    @doc_inherit
    def _log_metrics(self,
                     metrics: Dict[str, List[Dict[str, float]]],
                     model_id: str):
        # UAS
        kwargs = dict(metrics=metrics, metric_name='uas')
        uas_train_mean, uas_train_std_err = self._get_stats(split='train',
                                                            **kwargs)
        uas_test_mean, uas_test_std_err = self._get_stats(split='test',
                                                          **kwargs)

        # LAS
        kwargs = dict(metrics=metrics, metric_name='las')
        las_train_mean, las_train_std_err = self._get_stats(split='train',
                                                            **kwargs)
        las_test_mean, las_test_std_err = self._get_stats(split='test',
                                                          **kwargs)

        # Multiply scores by x100 to make them easier to read
        uas_train_mean *= 100
        uas_test_mean *= 100
        uas_train_std_err *= 100
        uas_test_std_err *= 100
        las_train_mean *= 100
        las_test_mean *= 100
        las_train_std_err *= 100
        las_test_std_err *= 100

        if not np.isnan(uas_train_std_err):
            uas_msg = (f'Mean UAS on the DEP part of DDT for {model_id}:\n'
                       f'  - Train: {uas_train_mean:.2f} +- '
                       f'{uas_train_std_err:.2f}\n'
                       f'  - Test: {uas_test_mean:.2f} +- '
                       f'{uas_test_std_err:.2f}')
            las_msg = (f'Mean LAS on the DEP part of DDT for {model_id}:\n'
                       f'  - Train: {las_train_mean:.2f} +- '
                       f'{las_train_std_err:.2f}\n'
                       f'  - Test: {las_test_mean:.2f} +- '
                       f'{las_test_std_err:.2f}')
        else:
            uas_msg = (f'UAS on the DEP part of DDT for {model_id}:\n'
                       f'  - Train: {uas_train_mean:.2f}\n'
                       f'  - Test: {uas_test_mean:.2f}')
            las_msg = (f'LAS on the DEP part of DDT for {model_id}:\n'
                       f'  - Train: {las_train_mean:.2f}\n'
                       f'  - Test: {las_test_mean:.2f}')

        logger.info(uas_msg)
        logger.info(las_msg)

    @doc_inherit
    def _get_spacy_token_labels(self, processed) -> List[Tuple[str, str]]:
        def get_heads_and_deps(token) -> Tuple[str, str]:
            dep = token.dep_.lower()
            if dep == 'root':
                head = '0'
            else:
                head = str(token.head.i + 1)
            return (head, dep)
        return [get_heads_and_deps(token) for token in processed]
