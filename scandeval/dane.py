'''NER evaluation of a language model on the DaNE dataset'''

from datasets import Dataset
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

from .token_classification import TokenClassificationBenchmark
from .datasets import load_dane
from .utils import doc_inherit


logger = logging.getLogger(__name__)


class DaneBenchmark(TokenClassificationBenchmark):
    '''Benchmark of language models on the DaNE dataset.

    Args:
        include_misc_tags (bool, optional):
            Whether to include MISC tags or not. If they are not included then
            they will be replaced by O tags, both in the dataset and in
            predictions. Defaults to True.
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
        include_misc_tags (bool): Whether MISC tags should be included.
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
                 include_misc_tags: bool = True,
                 cache_dir: str = '.benchmark_models',
                 evaluate_train: bool = False,
                 verbose: bool = False):
        self.include_misc_tags = include_misc_tags
        id2label = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER',
                    'I-PER', 'B-MISC', 'I-MISC', 'O']
        name = 'DaNE' if self.include_misc_tags else 'DaNE without MISC tags'
        super().__init__(metric_names=dict(micro_f1='Micro-average F1-score'),
                         name=name,
                         id2label=id2label,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_dane()
        train_dict = dict(doc=X_train['doc'],
                          tokens=X_train['tokens'],
                          orig_labels=y_train['ner_tags'])
        test_dict = dict(doc=X_test['doc'],
                         tokens=X_test['tokens'],
                         orig_labels=y_test['ner_tags'])
        train = Dataset.from_dict(train_dict)
        test = Dataset.from_dict(test_dict)
        return train, test

    @staticmethod
    def _remove_misc_tags(examples: dict) -> dict:
        examples['orig_labels'] = [['O' if label[-4:] == 'MISC' else label
                                    for label in label_list]
                                   for label_list in examples['orig_labels']]
        return examples

    @doc_inherit
    def _preprocess_data(self,
                         dataset: Dataset,
                         framework: str,
                         **kwargs) -> Dataset:
        if not self.include_misc_tags:
            dataset = dataset.map(self._remove_misc_tags, batched=True)
        preprocessed = super()._preprocess_data(dataset, framework, **kwargs)
        return preprocessed

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

            # Remove MISC labels if MISC tags are not included
            if not self.include_misc_tags:
                for i, prediction_list in enumerate(predictions):
                    for j, ner_tag in enumerate(prediction_list):
                        if ner_tag[-4:] == 'MISC':
                            predictions[i][j] = 'O'

        results = self._metric.compute(predictions=predictions,
                                       references=labels)
        return dict(micro_f1=results["overall_f1"])

    @doc_inherit
    def _get_spacy_token_labels(self, processed) -> List[str]:
        def get_ent(token) -> str:
            '''Helper function that extracts the entity from a SpaCy token'''

            # Replace predicted MISC tags with O tags if MISC tags are not
            # included
            if token.ent_type_ == 'MISC' and not self.include_misc_tags:
                return 'O'

            # Deal with the O tag separately, as it is the only tag not of the
            # form B-tag or I-tag
            elif token.ent_iob_ == 'O':
                return 'O'

            # In general return a tag of the form B-tag or I-tag
            else:
                return f'{token.ent_iob_}-{token.ent_type_}'

        return [get_ent(token) for token in processed]
