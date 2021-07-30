'''Abstract base class for evaluating models'''

from abc import ABC, abstractmethod
from datasets import Dataset
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers import (PreTrainedTokenizerBase,
                          AutoTokenizer,
                          AutoConfig,
                          TrainingArguments,
                          Trainer,
                          PrinterCallback)
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import requests
from bs4 import BeautifulSoup
import subprocess
import spacy
from tqdm.auto import tqdm
from collections import defaultdict
import copy

from .utils import block_terminal_output, MODEL_CLASSES, is_module_installed


block_terminal_output()


class Evaluator(ABC):
    '''Abstract base class for evaluating models.

    Args:
        TODO

    Parameters:
        TODO
    '''
    def __init__(self,
                 task: str,
                 num_labels: Optional[int] = None,
                 label2id: Optional[Dict[str, int]] = None,
                 cache_dir: str = '~/.cache/huggingface',
                 learning_rate: float = 2e-5,
                 epochs: int = 5,
                 warmup_steps: int = 50,
                 batch_size: int = 16):
        self.task = task
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.label2id = label2id
        if self.label2id is not None:
            self.id2label = {id: label for label, id in label2id.items()}
        else:
            self.id2label = None

    def _get_model_class(self, framework: str) -> _BaseAutoModelClass:
        return MODEL_CLASSES[framework][self.task]

    @staticmethod
    def _get_stats(metrics: Dict[str, List[Dict[str, float]]],
                   metric_name: str,
                   split: str) -> Tuple[float, float]:
        '''Helper function to compute the mean with confidence intervals.

        Args:
            metrics (dict):
                Dictionary with the names of the metrics as keys, of the form
                "<split>_<metric_name>", such as "val_f1", and values the
                metric values.
            metric_name (str):
                The name of the metric. Is used to collect the correct metric
                from `metrics`.
            split (str):
                The dataset split we are calculating statistics of. Is used to
                collect the correct metric from `metrics`.

        Returns:
            pair of floats:
                The mean micro-average F1-score and the radius of its 95%
                confidence interval.
        '''
        key = f'{split}_{metric_name}'
        metric_values = [dct[key] for dct in metrics[split]]
        mean = np.mean(metric_values)
        sample_std = np.std(metric_values, ddof=1)
        std_err = sample_std / np.sqrt(len(metric_values))
        return mean, 1.96 * std_err

    def _load_model(self,
                    model_id: str,
                    framework: Optional[str] = None) -> Dict[str, Any]:
        '''Load the model.

        Args:
            model_id (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.
            framework (str or None, optional):
                The framework the model has been built in. Currently supports
                'pytorch', 'tensorflow', 'jax' and 'spacy'. If None then this
                will be inferred from `model_id`. Defaults to None.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the
                value being the model. Can contain other objects related to the
                model, such as its tokenizer.

        Raises:
            RuntimeError: If the framework is not recognized.
        '''
        # Get the name of a framework supported for the model_id
        if framework is None:
            framework = self._fetch_model_metadata(model_id)['framework']

        if framework in ['pytorch', 'tensorflow', 'jax']:
            config = AutoConfig.from_pretrained(model_id,
                                                num_labels=self.num_labels,
                                                label2id=self.label2id,
                                                id2label=self.id2label)

            model_cls = self._get_model_class(framework=framework)
            model = model_cls.from_pretrained(model_id,
                                              config=config,
                                              cache_dir=self.cache_dir)

            # If the model is a subclass of a RoBERTa model then we have to add
            # a prefix space to the tokens, by the way the model is
            # constructed.
            prefix = 'RobertaModel' in type(model).__name__
            tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                      use_fast=True,
                                                      add_prefix_space=prefix)

            return dict(model=model, tokenizer=tokenizer)

        elif framework == 'spacy':
            local_model_id = model_id.split('/')[-1]

            # Download the model if it has not already been so
            if not is_module_installed(local_model_id):
                url = (f'https://huggingface.co/{model_id}/resolve/main/'
                       f'{local_model_id}-any-py3-none-any.whl')
                subprocess.check_output(['pip3', 'install', url])

            # Load the model
            model = spacy.load(local_model_id)

            return dict(model=model)

        else:
            raise RuntimeError(f'The framework "{framework}" is not '
                               f'supported!')

    @abstractmethod
    def _load_data(self) -> Tuple[Dataset, Dataset]:
        '''Load the datasets.

        Returns:
            A triple of HuggingFace datasets:
                The train and test datasets.
        '''
        pass

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, **kwargs) -> Dataset:
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
                The first array contains the probability predictions and the
                second array contains the true labels.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the
                metric values as values.
        '''
        pass

    @abstractmethod
    def _log_metrics(self, metrics: Dict[str, List[Dict[str, float]]]):
        '''Log the metrics.

        Args:
            metrics (dict):
                The metrics that are to be logged. This is a dict with keys
                'train' and 'test', with values being lists of dictionaries
                full of metrics.
        '''
        pass

    @abstractmethod
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset) -> tuple:
        '''Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model): The model.
            dataset (HuggingFace dataset): The dataset.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the
                second array contains the true labels.
        '''
        pass

    @staticmethod
    def _fetch_model_metadata(model_id: str) -> Dict[str, str]:
        '''Fetches metdataof a model from the HuggingFace Hub.

        Args:
            model_id (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.

        Returns:
            dict:
                The keys are names of metadata, with the values being the
                strings that describe the value of that metadata. Keys involve
                'framework' and 'task', where a framework could be 'pytorch'
                and a task could be 'token-classification'.

        Raises:
            RuntimeError: If the extracted framework is not recognized.
        '''
        # Parse all the anchor tags from the model website
        url = 'https://www.huggingface.co/' + model_id
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        a_tags = soup.find_all('a')
        a_tags_with_class = [a for a in a_tags if a.get('class') is not None]

        # Fetch the frameworks from the model website
        frameworks = [a['tag-id'] for a in a_tags_with_class
                                  if 'tag-red' in a['class']]

        # Extract a single valid framework in which the model has been
        # implemented
        valid_frameworks = ['pytorch', 'tensorflow', 'jax', 'spacy']
        for valid_framework in valid_frameworks:
            if valid_framework in frameworks:
                framework = valid_framework
                break
        else:
            raise RuntimeError(f'Cannot detect the framework of {model_id}!')

        # Fetch the model tasks from the model website
        tasks = [a['tag-id'] for a in a_tags_with_class
                             if 'tag-white' in a['class']]

        # Extract a single valid task on which the model has been trained
        task = tasks[0]

        return dict(framework=framework, task=task)

    def __call__(self,
                 model_id: str,
                 num_finetunings: int = 10,
                 progress_bar: bool = True
                 ) -> Dict[str, List[Dict[str, float]]]:
        '''Finetune and evaluate a model.

        Args:
            model_id (str):
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
                The keys in the dict are 'train' and 'test', and the values
                contain the metrics for that given dataset split.

        Raises:
            RuntimeError: If the extracted framework is not recognized.
        '''
        # Load the model and its metadata
        model_metadata = self._fetch_model_metadata(model_id)
        framework = model_metadata['framework']
        task = model_metadata['task']
        model_dict = self._load_model(model_id, framework=framework)

        # Load the dataset
        train, test = self._load_data()

        # Set up progress bar
        if progress_bar:
            desc = 'Evaluating'
            itr = tqdm(range(num_finetunings), desc=desc)
        else:
            itr = range(num_finetunings)

        if framework in ['pytorch', 'tensorflow', 'jax']:
            # Extract the model and tokenizer
            model = model_dict['model']
            tokenizer = model_dict['tokenizer']

            # Preprocess the datasets
            preprocessed_train = self._preprocess_data(train,
                                                       framework=framework,
                                                       tokenizer=tokenizer)
            preprocessed_test = self._preprocess_data(test,
                                                      framework=framework,
                                                      tokenizer=tokenizer)

            # Load the data collator
            data_collator = self._load_data_collator(tokenizer)

            # Initialise training arguments
            training_args = TrainingArguments(
                output_dir='.',
                evaluation_strategy='no',
                logging_strategy='no',
                save_strategy='no',
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=1,
                learning_rate=self.learning_rate,
                num_train_epochs=self.epochs,
                warmup_steps=self.warmup_steps,
                report_to='all',
                save_total_limit=0,
                log_level='error',
                log_level_replica='error'
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
                                  tokenizer=tokenizer,
                                  data_collator=data_collator,
                                  compute_metrics=self._compute_metrics)

                # Remove the callback which prints the metrics after each
                # evaluation
                trainer.remove_callback(PrinterCallback)

                # Finetune the model
                if task == 'fill-mask':
                    trainer.train()

                # Log training metrics and save the state
                train_metrics = trainer.evaluate(preprocessed_train,
                                                 metric_key_prefix='train')
                metrics['train'].append(train_metrics)

                # Log test metrics
                test_metrics = trainer.evaluate(preprocessed_test,
                                                metric_key_prefix='test')
                metrics['test'].append(test_metrics)

            self._log_metrics(metrics)
            return metrics

        elif framework == 'spacy':
            # Load the model
            model = model_dict['model']

            # Preprocess the datasets
            preprocessed_train = self._preprocess_data(train,
                                                       framework=framework)
            preprocessed_test = self._preprocess_data(test,
                                                      framework=framework)

            train_preds_labels = self._get_spacy_predictions_and_labels(
                model=model, dataset=preprocessed_train
            )
            test_preds_labels = self._get_spacy_predictions_and_labels(
                model=model, dataset=preprocessed_test
            )

            metrics = dict(train=self._compute_metrics(train_preds_labels),
                           test=self._compute_metrics(test_preds_labels))

            self._log_metrics(metrics)
            return metrics

        else:
            raise RuntimeError(f'The framework "{framework}" is not '
                               f'supported!')
