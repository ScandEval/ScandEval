'''NER evaluation of a language model on the DaNE dataset'''

from datasets import Dataset
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm.auto import tqdm
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
                 include_misc_tags: bool = True,
                 verbose: bool = False):
        self.include_misc_tags = include_misc_tags
        label2id = {'B-LOC': 0,
                    'I-LOC': 1,
                    'B-ORG': 2,
                    'I-ORG': 3,
                    'B-PER': 4,
                    'I-PER': 5,
                    'B-MISC': 6,
                    'I-MISC': 7,
                    'O': 8}
        super().__init__(num_labels=9,
                         epochs=5,
                         label2id=label2id,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
                         verbose=verbose)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        X_train, X_test, y_train, y_test = load_dane()
        train = Dataset.from_dict(dict(docs=X_train, orig_labels=y_train))
        test = Dataset.from_dict(dict(docs=X_test, orig_labels=y_test))
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
        kwargs = dict(metrics=metrics, metric_name='micro_f1')
        train_mean, train_std_err = self._get_stats(split='train', **kwargs)
        test_mean, test_std_err = self._get_stats(split='test', **kwargs)

        # Multiply scores by x100 to make them easier to read
        train_mean *= 100
        test_mean *= 100
        train_std_err *= 100
        test_std_err *= 100

        if not np.isnan(train_std_err):
            msg = (f'Mean micro-average F1-scores on DaNE for {model_id}:\n'
                   f'  - Train: {train_mean:.2f} +- {train_std_err:.2f}\n'
                   f'  - Test: {test_mean:.2f} +- {test_std_err:.2f}')
        else:
            msg = (f'Micro-average F1-scores on DaNE for {model_id}:\n'
                   f'  - Train: {train_mean:.2f}\n'
                   f'  - Test: {test_mean:.2f}')

        logger.info(msg)

    @staticmethod
    def _extract_spacy_predictions(tokens_processed: tuple) -> dict:
        tokens, processed = tokens_processed

        # Get the model's named entity predictions
        ner_tags = {ent.text: ent.label_ for ent in processed.ents}

        # Organise the predictions to make them comparable to the labels
        preds = list()
        for token in tokens:
            for ner_tag in ner_tags.keys():
                if ner_tag.startswith(token):
                    preds.append('B-' + ner_tags[ner_tag])
                    break
                elif token in ner_tag:
                    preds.append('I-' + ner_tags[ner_tag])
                    break
            else:
                preds.append('O')

        return preds

    @doc_inherit
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        # Initialise progress bar
        if progress_bar:
            itr = tqdm(dataset['doc'])
        else:
            itr = dataset['doc']

        disable = ['tok2vec', 'tagger', 'parser',
                   'attribute_ruler', 'lemmatizer']
        processed = model.pipe(itr, disable=disable, batch_size=32)

        map_fn = self._extract_spacy_predictions
        predictions = map(map_fn, zip(dataset['docs'], processed))

        return list(predictions), dataset['orig_labels']
