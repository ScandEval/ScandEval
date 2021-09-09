'''NER evaluation of a language model on the Nynorsk part of NorNE'''

import logging

from .abstract import NerBenchmark


logger = logging.getLogger(__name__)


class NorneNNBenchmark(NerBenchmark):
    '''Benchmark of language models on the Nynorsk part of the NorNE dataset.

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
        super().__init__(name='norne-nn',
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         verbose=verbose)
