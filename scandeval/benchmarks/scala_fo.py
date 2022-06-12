'''Correct spelling classification on the Faroese part of the ScaLA dataset'''

import logging

from .abstract import TextClassificationBenchmark


logger = logging.getLogger(__name__)


class ScalaFOBenchmark(TextClassificationBenchmark):
    '''Benchmark of language models on the Faroese part of the ScaLA dataset.

    Args:
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        evaluate_train (bool, optional):
            Whether the models should be evaluated on the training scores.
            Defaults to False.
        use_auth_token (bool, optional):
            Whether the benchmark should use an authentication token. Defaults
            to False.
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
        use_auth_token (bool): Whether an authentication token should be used.
        verbose (bool): Whether to print additional output.
    '''
    def __init__(self,
                 cache_dir: str = '.benchmark_models',
                 evaluate_train: bool = False,
                 use_auth_token: bool = False,
                 verbose: bool = False):
        id2label = ['incorrect', 'correct']
        label_synonyms = [
            ['LABEL_0', id2label[0]],
            ['LABEL_1', id2label[1]],
        ]
        super().__init__(name='scala-fo',
                         language='fo',
                         id2label=id2label,
                         label_synonyms=label_synonyms,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         use_auth_token=use_auth_token,
                         verbose=verbose)
