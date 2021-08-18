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
        id2label = id2label_head + id2label_dep
        super().__init__(name='the DEP part of DDT',
                         metric_names=dict(las='LAS', uas='UAS'),
                         id2label=id2label,
                         cache_dir=cache_dir,
                         two_labels=True,
                         split_point=len(id2label_head),
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_ddt_dep()

        train_labels = [list(zip(head, dep))
                        for head, dep in zip(y_train['heads'],
                                             y_train['deps'])]
        test_labels = [list(zip(head, dep))
                       for head, dep in zip(y_test['heads'],
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

        # If `id2label` is given then assume that `predictions` contain ID
        # logits for every token, where an ID can mean both a head and a dep,
        # so it needs to be split up in these two halves.
        if id2label is not None:

            # Here we split up the predictions into the "head part" and the
            # "dep part"
            predictions1 = predictions[:, :, :self.split_point]
            predictions2 = predictions[:, :, self.split_point:]

            # With the predictions split up, we can then get the highest logits
            # to get the head and dep label
            raw_predictions1 = np.argmax(predictions1, axis=-1)
            raw_predictions2 = np.argmax(predictions2, axis=-1)

            # The `labels` are assumed to be of shape
            # (batch_size, sequence_length, label_type), where `label_type` is
            # a binary number indicating either the head label or the dep
            # label. Here we extract the two different labels.
            labels1 = labels[:, :, 0]
            labels2 = labels[:, :, 1]
            labels2 = np.where(labels2 > 0, labels2-self.split_point, labels2)

            # Remove ignored indices from predictions and labels
            predictions1 = [
                [id2label[pred] for pred, lbl in zip(prediction, label)
                 if lbl != -100]
                for prediction, label in zip(raw_predictions1, labels1)
            ]
            predictions2 = [
                [id2label[pred] for pred, lbl in zip(prediction, label)
                 if lbl != -100]
                for prediction, label in zip(raw_predictions2, labels2)
            ]
            labels1 = [[id2label[lbl] for lbl in label if lbl != -100]
                       for label in labels1]
            labels2 = [[id2label[lbl] for lbl in label if lbl != -100]
                       for label in labels2]

            # Next merge the predictions and labels, so that we have a pair of
            # predicted/gold labels for each token
            predictions_merged = [list(map(str, zip(head, dep)))
                                  for head, dep in zip(predictions1,
                                                       predictions2)]
            labels_merged = [list(map(str, zip(head, dep)))
                             for head, dep in zip(labels1, labels2)]

        # If `id2label` is not given then assume that the predictions and
        # labels contain a pair (head, dep) for every token.
        else:
            # Convert the pair of labels to a single one by converting it into
            # strings. This is used in LAS computations.
            predictions_merged = [list(map(str, tuples))
                                  for tuples in predictions]
            labels_merged = [list(map(str, tuples)) for tuples in labels]

            # Extract the heads predictions and labels, used in UAS computation
            predictions1 = [[head for head, _ in preds]
                            for preds in predictions]
            labels1 = [[head for head, _ in label_list]
                       for label_list in labels]

        # Compute metrics for the heads, which is used in UAS computation
        results_head = self._metric.compute(predictions=predictions1,
                                            references=labels1)

        # Compute metrics for the merged heads and deps, which is used in LAS
        # computation
        results_merged = self._metric.compute(predictions=predictions_merged,
                                              references=labels_merged)

        # Extract UAS and LAS and return them
        uas = results_head['overall_accuracy']
        las = results_merged['overall_accuracy']
        return dict(uas=uas, las=las)

    @doc_inherit
    def _get_spacy_token_labels(self, processed) -> List[str]:
        def get_heads_and_deps(token) -> List[str]:
            dep = token.dep_.lower().replace('obl:loc', 'obl:lmod')
            if dep == 'root':
                head = '0'
            else:
                head = str(token.head.i + 1)
            return [head, dep]
        return [get_heads_and_deps(token) for token in processed]
