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
                 include_misc_tags: bool = True,
                 evaluate_train: bool = False,
                 verbose: bool = False):
        self.include_misc_tags = include_misc_tags
        id2label = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER',
                    'I-PER', 'B-MISC', 'I-MISC', 'O']
        super().__init__(epochs=10,
                         warmup_steps=38,
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
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

            # Remove MISC labels
            if not self.include_misc_tags:
                for i, prediction_list in enumerate(predictions):
                    for j, ner_tag in enumerate(prediction_list):
                        if ner_tag[-4:] == 'MISC':
                            predictions[i][j] = 'O'

        results = self._metric.compute(predictions=predictions,
                                       references=labels)
        return dict(micro_f1=results["overall_f1"])

    @doc_inherit
    def _log_metrics(self,
                     metrics: Dict[str, List[Dict[str, float]]],
                     model_id: str):
        scores = self._get_stats(metrics, 'micro_f1')
        test_score, test_se = scores['test']
        test_score *= 100
        test_se *= 100

        if self.include_misc_tags:
            misc_txt = 'without MISC tags'
        else:
            misc_txt = ''

        msg = (f'Micro-average F1-scores on DaNE {misc_txt} for {model_id}:\n'
               f'  - Test: {test_score:.2f} ± {test_se:.2f}')

        if 'train' in scores.keys():
            train_score, train_se = scores['train']
            train_score *= 100
            train_se *= 100
            msg += f'\n  - Train: {train_score:.2f} ± {train_se:.2f}'

        logger.info(msg)

    @doc_inherit
    def _get_spacy_token_labels(self, processed) -> List[str]:
        def get_ent(token) -> str:
            if token.ent_iob_ == 'O':
                return 'O'
            else:
                return f'{token.ent_iob_}-{token.ent_type_}'

        return [get_ent(token) for token in processed]
