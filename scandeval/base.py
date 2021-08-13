'''Abstract base class for evaluating models'''

from abc import ABC, abstractmethod
from datasets import Dataset
from transformers.models.auto.auto_factory import _BaseAutoModelClass
import transformers.utils.logging as tf_logging
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
from tqdm.auto import tqdm
from collections import defaultdict
import warnings
from functools import partial
import gc

from .utils import (MODEL_CLASSES, is_module_installed, InvalidBenchmark,
                    TwolabelTrainer)


class BaseBenchmark(ABC):
    '''Abstract base class for evaluating models.

    Args:
        task (str):
            The type of task to be benchmarked.
        num_labels (int or None, optional):
            The number of labels in the dataset. Defaults to None.
        label2id (dict or None, optional):
            A dictionary that converts labels to their indices. This will only
            be used if the pretrained model does not already have one. Defaults
            to None.
        evaluate_train (bool, optional):
            Whether the models should be evaluated on the training scores.
            Defaults to False.
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        learning_rate (float, optional):
            What learning rate to use when finetuning the models. Defaults to
            2e-5.
        epochs (int, optional):
            The number of epochs to finetune for. Defaults to 5.
        warmup_steps (int, optional):
            The number of training steps in which the learning rate will be
            warmed up, meaning starting from nearly 0 and progressing up to
            `learning_rate` after `warmup_steps` many steps. Defaults to 50.
        batch_size (int, optional):
            The batch size used while finetuning. Must be a multiple of 2, and
            at most 32. Defaults to 32.
        verbose (bool, optional):
            Whether to print additional output during evaluation. Defaults to
            False.

    Parameters:
        task (str): The type of task to be benchmarked.
        num_labels (int or None): The number of labels in the dataset.
        label2id (dict or None): A dictionary converting labels to indices.
        id2label (dict or None): A dictionary converting indices to labels.
        cache_dir (str): Directory where models are cached.
        learning_rate (float): Learning rate used while finetuning.
        epochs (int): The number of epochs to finetune for.
        warmup_steps (int): Number of steps used to warm up the learning rate.
        batch_size (int): The batch size used while finetuning.
        verbose (bool): Whether to print additional output.

    Raises:
        TypeError:
            If `batch_size` is not among 1, 2, 4, 8, 16 or 32.
    '''
    def __init__(self,
                 task: str,
                 num_labels: Optional[int] = None,
                 id2label: Optional[List[str]] = None,
                 evaluate_train: bool = False,
                 cache_dir: str = '.benchmark_models',
                 learning_rate: float = 2e-5,
                 epochs: int = 5,
                 warmup_steps: int = 50,
                 batch_size: int = 32,
                 multilabel: bool = False,
                 split_point: Optional[int] = None,
                 verbose: bool = False):
        if batch_size not in [1, 2, 4, 8, 16, 32]:
            raise TypeError('The batch size must be either 1, 2, 4, 8, '
                            '16 or 32.')
        self.batch_size = batch_size
        self.gradient_accumulation = 32 // batch_size
        self.evaluate_train = evaluate_train
        self.task = task
        self.cache_dir = cache_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.num_labels = num_labels
        self.id2label = id2label
        self.multilabel = multilabel
        self.split_point = split_point
        self.verbose = verbose
        if self.id2label is not None:
            self.label2id = {label: id for id, label in enumerate(id2label)}
        else:
            self.label2id = None
        if verbose:
            tf_logging.set_verbosity_warning()

    def _get_model_class(self, framework: str) -> _BaseAutoModelClass:
        return MODEL_CLASSES[framework][self.task]

    @staticmethod
    def _get_stats(metrics: Dict[str, List[Dict[str, float]]],
                   metric_name: str) -> Dict[str, Tuple[float, float]]:
        '''Helper function to compute the mean with confidence intervals.

        Args:
            metrics (dict):
                Dictionary with the names of the metrics as keys, of the form
                "<split>_<metric_name>", such as "val_f1", and values the
                metric values.
            metric_name (str):
                The name of the metric. Is used to collect the correct metric
                from `metrics`.

        Returns:
            dict:
                Dictionary with keys among 'train' and 'test', with
                corresponding values being a pair of floats, containing the
                score and the radius of its 95% confidence interval.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            results = dict()

            if 'train' in metrics.keys():
                train_scores = [dct[f'train_{metric_name}']
                                for dct in metrics['train']]
                train_score = train_scores[0]

                if len(train_scores) > 1:
                    sample_std = np.std(train_scores, ddof=1)
                    train_se = sample_std / np.sqrt(len(train_scores))
                else:
                    train_se = np.nan

                results['train'] = (train_score, 1.96 * train_se)

            if 'test' in metrics.keys():
                test_scores = [dct[f'test_{metric_name}']
                               for dct in metrics['test']]
                test_score = test_scores[0]

                if len(test_scores) > 1:
                    sample_std = np.std(test_scores, ddof=1)
                    test_se = sample_std / np.sqrt(len(test_scores))
                else:
                    test_se = np.nan

                results['test'] = (test_score, 1.96 * test_se)

            return results

    def _load_model(self,
                    model_id: str,
                    framework: Optional[str] = None,
                    task: Optional[str] = None) -> Dict[str, Any]:
        '''Load the model.

        Args:
            model_id (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.
            framework (str or None, optional):
                The framework the model has been built in. Currently supports
                'pytorch', 'tensorflow', 'jax' and 'spacy'. If None then this
                will be inferred from `model_id`. Defaults to None.
            task (str or None, optional):
                The task for which the model was trained on. If None then this
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
        if framework is None or task is None:
            model_metadata = self._fetch_model_metadata(model_id)
            if framework is None:
                framework = model_metadata['framework']
            if task is None:
                task = model_metadata['task']

        # Ensure that the framework is installed
        try:
            if framework == 'pytorch':
                import torch  # noqa
            elif framework == 'tensorflow':
                import tensorflow  # noqa
            elif framework == 'jax':
                import flax  # noqa
            elif framework == 'spacy':
                import spacy

                # Ignore warnings from SpaCy. This has to be called after the
                # import, as the __init__.py file of SpaCy sets the warning
                # levels of SpaCy warning W036
                import warnings
                warnings.filterwarnings('ignore', module='spacy*')

        except ModuleNotFoundError:
            msg = (f'The model {model_id} is built using the {framework} '
                   f'framework which is not installed. Try installing the '
                   f'ScandEval package as `pip install '
                   f'scandeval[{framework}]`.')
            raise ModuleNotFoundError(msg)

        if framework in ['pytorch', 'tensorflow', 'jax']:

            if task == 'fill-mask':
                params = dict(num_labels=self.num_labels,
                              label2id=self.label2id,
                              id2label=self.id2label)
            else:
                params = dict()

            try:
                config = AutoConfig.from_pretrained(model_id, **params)

                model_cls = self._get_model_class(framework=framework)

                model = model_cls.from_pretrained(model_id,
                                                  config=config,
                                                  cache_dir=self.cache_dir)
            except (OSError, ValueError):
                raise InvalidBenchmark(f'The model {model_id} could not be '
                                       f'loaded from the HuggingFace hub.')

            # If the model is a subclass of a RoBERTa model then we have to add
            # a prefix space to the tokens, by the way the model is
            # constructed.
            prefix = 'Roberta' in type(model).__name__
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
                subprocess.run(['pip3', 'install', url])

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
        pass

    @abstractmethod
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
    def _log_metrics(self,
                     metrics: Dict[str, List[Dict[str, float]]],
                     model_id: str):
        '''Log the metrics.

        Args:
            metrics (dict):
                The metrics that are to be logged. This is a dict with keys
                'train' and 'test', with values being lists of dictionaries
                full of metrics.
            model_id (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.
        '''
        pass

    @abstractmethod
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
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

        # Extract a single valid task on which the model has been trained. If
        # no task has been specified on the model card then assume that it is
        # 'fill-mask'
        task = tasks[0] if len(tasks) > 0 else 'fill-mask'

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
        model_dict = self._load_model(model_id, **model_metadata)

        # Load the dataset
        train, test = self._load_data()

        # Initialise random number generator
        rng = np.random.default_rng(4242)

        # Get bootstrap sample indices
        if task == 'fill-mask' or self.evaluate_train:
            train_bidxs = rng.integers(0, len(train),
                                       size=(num_finetunings, len(train)))
        test_bidxs = rng.integers(0, len(test), size=(10, len(test)))

        if framework in ['pytorch', 'tensorflow', 'jax']:

            # Extract the model and tokenizer
            model = model_dict['model']
            tokenizer = model_dict['tokenizer']

            # Preprocess the datasets
            try:
                params = dict(framework=framework,
                              config=model.config,
                              tokenizer=tokenizer)
                if task == 'fill-mask' or self.evaluate_train:
                    train = self._preprocess_data(train, **params)
                test = self._preprocess_data(test, **params)
            except ValueError:
                raise InvalidBenchmark('Preprocessing of the dataset could '
                                       'not be done.')

            # Get bootstrapped datasets
            if task == 'fill-mask' or self.evaluate_train:
                trains = [train]
                trains += [Dataset.from_dict(train[train_bidxs[idx]])
                           for idx in range(num_finetunings - 1)]
            tests = [test]
            tests += [Dataset.from_dict(test[test_bidxs[idx]])
                      for idx in range(test_bidxs.shape[0])]

            # Set up progress bar
            if task == 'fill-mask':
                if progress_bar:
                    itr = tqdm(range(num_finetunings))
                else:
                    itr = range(num_finetunings)
            else:
                itr = [0]

            # Load the data collator
            data_collator = self._load_data_collator(tokenizer)

            # Enable `transformers` verbosity to see a training
            # progress bar
            if progress_bar:
                tf_logging.set_verbosity_warning()

            # Initialise training arguments
            training_args = TrainingArguments(
                output_dir='.',
                evaluation_strategy='no',
                logging_strategy='no',
                save_strategy='no',
                report_to='none',
                save_total_limit=0,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                num_train_epochs=self.epochs,
                warmup_steps=self.warmup_steps,
                gradient_accumulation_steps=self.gradient_accumulation
            )

            # Disable `transformers` verbosity again
            if not self.verbose:
                tf_logging.set_verbosity_error()

            metrics = defaultdict(list)
            for idx in itr:
                while True:
                    try:
                        # Reinitialise a new model
                        model = self._load_model(model_id,
                                                 **model_metadata)['model']

                        # Initialise compute_metrics function
                        compute_metrics = partial(
                            self._compute_metrics,
                            id2label=model.config.id2label
                        )

                        # Initialise Trainer
                        trainer_args = dict(model=model,
                                            args=training_args,
                                            train_dataset=trains[idx],
                                            tokenizer=tokenizer,
                                            data_collator=data_collator,
                                            compute_metrics=compute_metrics)
                        if self.multilabel:
                            trainer_args['split_point'] = self.split_point
                            trainer = TwolabelTrainer(**trainer_args)
                        else:
                            trainer = Trainer(**trainer_args)

                        # Remove the callback which prints the metrics after
                        # each evaluation
                        if not self.verbose:
                            trainer.remove_callback(PrinterCallback)

                        # Finetune the model
                        if task == 'fill-mask':
                            trainer.train()

                        # Log training metrics and save the state
                        if self.evaluate_train:
                            train_metrics = trainer.evaluate(
                                tests[idx],
                                metric_key_prefix='train'
                            )
                            metrics['train'].append(train_metrics)

                        # Log test metrics
                        for dataset in tests:
                            test_metrics = trainer.evaluate(
                                dataset,
                                metric_key_prefix='test'
                            )
                            metrics['test'].append(test_metrics)

                        break

                    except RuntimeError as e:
                        if not str(e).startswith('CUDA out of memory'):
                            raise InvalidBenchmark(str(e))
                        bs = training_args.per_device_train_batch_size
                        ga = training_args.gradient_accumulation_steps
                        if bs == 1:
                            raise InvalidBenchmark('CUDA out of memory, even '
                                                   'with a batch size of 1!')
                        training_args.per_device_train_batch_size = bs // 2
                        training_args.gradient_accumulation_steps = ga * 2
                        trainer.args = training_args

            self._log_metrics(metrics, model_id=model_id)

            # Garbage collection, to avoid memory issues
            del model, model_dict
            del train, test
            gc.collect()

            return metrics

        elif framework == 'spacy':
            # Load the model
            model = model_dict['model']

            # Preprocess the test datasets
            test = self._preprocess_data(test, framework=framework)
            tests = [test]
            tests += [Dataset.from_dict(test[test_bidxs[idx]])
                      for idx in range(test_bidxs.shape[0])]

            # Get the test predictions
            all_test_metrics = list()
            for dataset in tests:
                preds_labels = self._get_spacy_predictions_and_labels(
                    model=model,
                    dataset=dataset,
                    progress_bar=progress_bar
                )

                test_metrics = self._compute_metrics(preds_labels)
                test_metrics = {f'test_{key}': val
                                for key, val in test_metrics.items()}
                all_test_metrics.append(test_metrics)
            metrics = dict(test=all_test_metrics)

            # Preprocess the train datasets
            if self.evaluate_train:
                train = self._preprocess_data(train, framework=framework)
                trains = [train]
                trains += [Dataset.from_dict(train[train_bidxs[idx]])
                           for idx in range(num_finetunings - 1)]

            # Get the train predictions
            if self.evaluate_train:
                all_train_metrics = list()
                for dataset in trains:
                    preds_labels = self._get_spacy_predictions_and_labels(
                        model=model,
                        dataset=dataset,
                        progress_bar=progress_bar
                    )
                    train_metrics = self._compute_metrics(preds_labels)
                    train_metrics = {f'train_{key}': val
                                     for key, val in train_metrics.items()}
                all_train_metrics.append(train_metrics)
                metrics['train'] = all_train_metrics

            self._log_metrics(metrics, model_id=model_id)

            # Garbage collection, to avoid memory issues
            del model, model_dict
            gc.collect()

            return metrics

        else:
            raise RuntimeError(f'The framework "{framework}" is not '
                               f'supported!')
