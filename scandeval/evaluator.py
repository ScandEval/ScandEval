'''Abstract base class for evaluating models'''

from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import (PreTrainedTokenizerBase,
                          PreTrainedModel,
                          AutoTokenizer,
                          AutoConfig,
                          RobertaPreTrainedModel,
                          TrainingArguments,
                          Trainer,
                          PrinterCallback)
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import warnings
import datasets.utils.logging as ds_logging
import transformers.utils.logging as tf_logging
from tqdm.auto import tqdm
import copy
import numpy as np


# Ignore miscellaneous warnings
warnings.filterwarnings(
    'ignore',
    module='torch.nn.parallel*',
    message=('Was asked to gather along dimension 0, but all input '
             'tensors were scalars; will instead unsqueeze and return '
             'a vector.')
)
warnings.filterwarnings('ignore', module='seqeval*')

# Disable the tokenizer progress bars
ds_logging.get_verbosity = lambda: ds_logging.NOTSET

# Disable most of the `transformers` logging
tf_logging.set_verbosity_error()


class Evaluator(ABC):
    '''Abstract base class for evaluating models'''
    def __init__(self,
                 num_labels: int,
                 label2id: Dict[str, int],
                 prefer_flax: bool = False,
                 cache_dir: str = '~/.cache/huggingface',
                 learning_rate: float = 2e-5,
                 epochs: int = 5,
                 warmup_steps: int = 50,
                 batch_size: int = 16):
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = {id: label for label, id in label2id.items()}
        self.prefer_flax = prefer_flax
        self.cache_dir = cache_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size


    @abstractmethod
    def _get_model_class(self) -> type:
        '''Get the model class used for finetuning.

        Returns:
            type: The model class
        '''
        pass

    def _load_model(self,
                   transformer: str,
                   prefer_flax: Optional[bool] = None
                   ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        '''Load the model with its tokenizer.

        Args:
            transformer (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.
            prefer_flax (bool, optional):
                Whether to prefer to load the pretrained Flax model from the
                HuggingFace model repository. Defaults to False.

        Returns:
            tuple: The model and the tokenizer.
        '''
        if prefer_flax is None:
            prefer_flax = self.prefer_flax

        config = AutoConfig.from_pretrained(transformer,
                                            num_labels=self.num_labels,
                                            label2id=self.label2id,
                                            id2label=self.id2label)

        try:
            model = self._get_model_class().from_pretrained(transformer,
                                                     config=config,
                                                     from_flax=prefer_flax,
                                                     cache_dir=self.cache_dir)

        # Loading of model failed, due to the Flax/PyTorch version not being
        # available. Trying the other one.
        except OSError:
            prefer_flax = not prefer_flax
            model = self._get_model_class().from_pretrained(transformer,
                                                     config=config,
                                                     from_flax=prefer_flax,
                                                     cache_dir=self.cache_dir)

        # If the model is a subclass of `RobertaPreTrainedModel` then we have
        # to add a prefix space to the tokens, by the way the model is
        # constructed.
        prefix = isinstance(model, RobertaPreTrainedModel)
        tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True,
                                                  add_prefix_space=prefix)

        return model, tokenizer

    @abstractmethod
    def _load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        '''Load the datasets.

        Returns:
            A triple of HuggingFace datasets:
                The train, validation and test datasets.
        '''
        pass

    @abstractmethod
    def _preprocess_data(self,
                         dataset: Dataset,
                         tokenizer: PreTrainedTokenizerBase) -> Dataset:
        '''Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset (HuggingFace dataset):
                The dataset to preprocess.
            tokenizer (HuggingFace tokenizer):
                A pretrained tokenizer.

        Returns:
            HuggingFace dataset: The preprocessed dataset.
        '''
        pass

    @abstractmethod
    def _load_data_collator(self,
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
        pass

    @abstractmethod
    def _compute_metrics(self,
                         predictions_and_labels: tuple) -> Dict[str, float]:
        '''Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions, of shape
                (num_samples, sequence_length, num_classes), and the second
                array contains the true labels, of shape (num_samples,
                sequence_length).

        Returns:
            dict:
                A dictionary with key 'micro_f1' and the micro-average F1-score
                as value.
        '''
        pass

    @abstractmethod
    def _log_metrics(self, metrics: Dict[str, List[Dict[str, float]]]):
        '''Log the metrics.

        Args:
            metrics (dict):
                The metrics that are to be logged. This is a dict with keys
                'train', 'val' and 'split', with values being lists of
                dictionaries full of metrics.
        '''
        pass

    @staticmethod
    def get_stats(metrics: Dict[str, List[Dict[str, float]]],
                  metric_name: str,
                  split: str) -> Tuple[float, float]:
        '''Helper function to compute the mean with confidence intervals.

        Args:
            split (str):
                The dataset split we are calculating statistics of.

        Returns:
            pair of floats:
                The mean micro-average F1-score and the radius of its 95%
                confidence interval.
        '''
        key = f'{split}_{metric_name}'
        metric_values= [dct[key] for dct in metrics[split]]
        mean = np.mean(metric_values)
        sample_std = np.std(metric_values, ddof=1)
        std_err = sample_std / np.sqrt(len(metric_values))
        return mean, 1.96 * std_err

    def evaluate(self,
                 transformer: str,
                 num_finetunings: int = 10,
                 progress_bar: bool = True
                 ) -> Dict[str, List[Dict[str, float]]]:
        '''Finetune and evaluate a transformer model.

        Args:
            transformer (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.
            num_finetunings (int, optional):
                The number of times to finetune. These results will be used to
                calculate confidence intervals of the means of the metrics.
                Defaults to 10.
            progress_bar (bool, optional):
                Whether to show a progress bar or not. Defaults to True.

        Returns:
            dict:
                The keys in the dict are 'train', 'val' and 'test, and the
                values contain the metrics for that given dataset split.
        '''
        # Load the tokenizer and model
        model, tokenizer = self._load_model(transformer)

        # Set up progress bar
        if progress_bar:
            desc = 'Finetuning and evaluating'
            itr = tqdm(range(num_finetunings), desc=desc)
        else:
            itr = range(num_finetunings)

        # Load the dataset
        train, val, test = self._load_data()

        # Preprocess the datasets
        preprocessed_train = self._preprocess_data(train, tokenizer)
        preprocessed_val = self._preprocess_data(val, tokenizer)
        preprocessed_test = self._preprocess_data(test, tokenizer)

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        # Initialise training arguments
        training_args = TrainingArguments(
            output_dir='.',
            evaluation_strategy='no',
            logging_strategy='no',
            save_strategy='no',
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            report_to='all',
            save_total_limit=0,
            log_level='error',  # Separate logging levels for Trainer
            log_level_replica='error'  # Separate logging levels for Trainer
        )

        metrics = defaultdict(list)
        for _ in itr:

            # Make a copy of the original model, so that we do not continue
            # training the same one
            model_copy = copy.deepcopy(model)

            # Initialise Trainer
            trainer = Trainer(model=model_copy,
                              args=training_args,
                              train_dataset=preprocessed_train,
                              eval_dataset=preprocessed_val,
                              tokenizer=tokenizer,
                              data_collator=data_collator,
                              compute_metrics=self._compute_metrics)

            # Remove the callback which prints the metrics after each
            # evaluation
            trainer.remove_callback(PrinterCallback)

            # Finetune the model
            trainer.train()

            # Log training metrics and save the state
            train_metrics = trainer.evaluate(preprocessed_train,
                                             metric_key_prefix='train')
            metrics['train'].append(train_metrics)

            # Log validation metrics
            val_metrics = trainer.evaluate(preprocessed_val,
                                           metric_key_prefix='val')
            metrics['val'].append(val_metrics)

            # Log test metrics
            test_metrics = trainer.evaluate(preprocessed_test,
                                            metric_key_prefix='test')
            metrics['test'].append(test_metrics)

        self._log_metrics(metrics)
        return metrics
