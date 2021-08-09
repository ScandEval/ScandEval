'''Abstract text classification benchmark'''

from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from datasets import Dataset, load_metric
from functools import partial
import logging
from abc import ABC
from typing import Optional

from .base import BaseBenchmark
from .utils import InvalidBenchmark


logger = logging.getLogger(__name__)


class TextClassificationBenchmark(BaseBenchmark, ABC):
    '''Abstract text classification benchmark.

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
                 id2label: list,
                 epochs: int,
                 cache_dir: str = '.benchmark_models',
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 50,
                 batch_size: int = 16,
                 verbose: bool = False):
        self._metric = load_metric('f1')
        super().__init__(task='text-classification',
                         num_labels=len(id2label),
                         id2label=id2label,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         epochs=epochs,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size,
                         verbose=verbose)

    def _load_data_collator(
            self,
            tokenizer: Optional[PreTrainedTokenizerBase] = None):
        '''Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (HuggingFace tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not
                used in the initialisation of the data collator. Defaults to
                None.

        Returns:
            HuggingFace data collator: The data collator.
        '''
        return DataCollatorWithPadding(tokenizer, padding='longest')

    def create_numerical_labels(self, examples: dict, label2id: dict) -> dict:
        try:
            examples['label'] = [label2id[lbl]
                                 for lbl in examples['orig_label']]
        except KeyError:
            raise InvalidBenchmark(f'One of the labels in the dataset, '
                                   f'{examples["orig_label"]}, does not occur '
                                   f'in the label2id dictionary {label2id}.')
        return examples

    def _preprocess_data(self,
                         dataset: Dataset,
                         framework: str,
                         **kwargs) -> Dataset:
        '''Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset (HuggingFace dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in
                preprocessing the dataset.

        Returns:
            HuggingFace dataset: The preprocessed dataset.
        '''
        if framework in ['pytorch', 'tensorflow', 'jax']:
            tokenizer = kwargs['tokenizer']

            def tokenise(examples: dict) -> dict:
                doc = examples['doc']
                return tokenizer(doc, truncation=True, padding=True)
            tokenised = dataset.map(tokenise, batched=True)

            numericalise = partial(self.create_numerical_labels,
                                   label2id=kwargs['config'].label2id)
            preprocessed = tokenised.map(numericalise, batched=True)

            return preprocessed.remove_columns(['doc', 'orig_label'])

        elif framework == 'spacy':
            raise InvalidBenchmark('Evaluation of sentiment predictions '
                                   'for SpaCy models is not yet implemented.')
