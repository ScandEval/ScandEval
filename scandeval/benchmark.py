'''Fetches an updated list of all Scandinavian models on the HuggingFace Hub'''

import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Union, Dict
from collections import defaultdict
import logging
import json
from pathlib import Path

from .utils import InvalidBenchmark, get_all_datasets


logger = logging.getLogger(__name__)


class Benchmark:
    '''Benchmarking all the Scandinavian language models.

    Args:
        progress_bar (bool, optional):
            Whether progress bars should be shown. Defaults to True.
        save_results (bool, optional):
            Whether to save the benchmark results to
            'scandeval_benchmark_results.json'. Defaults to False.
        language (str or list of str, optional):
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this to
            'all' if all languages (also non-Scandinavian) should be considered.
            Defaults to ['da', 'sv', 'no'].
        model_language (str or list of str or None, optional):
            The language codes of the languages to include for models. If specified
            then this overrides the `language` parameter for model languages. Defaults
            to None.
        dataset_language (str or list of str or None, optional):
            The language codes of the languages to include for datasets. If specified
            then this overrides the `language` parameter for dataset languages. Defaults
            to None.
        task (str or list of str, optional):
            The tasks to consider in the list. Set this to 'all' if all
            tasks should be considered. Defaults to 'all'.
        evaluate_train (bool, optional):
            Whether to evaluate the training set as well. Defaults to False.
        raise_error_on_invalid_model (bool, optional):
            Whether to raise an error if a model is invalid. Defaults to
            False.
        train_size (list of int, optional):
            The training sizes to consider. Defaults to [1024].
        use_auth_token (bool, optional):
            Whether the benchmark should use an authentication token. Defaults
            to False.
        verbose (bool, optional):
            Whether to output additional output. Defaults to False.

    Attributes:
        progress_bar (bool): Whether progress bars should be shown.
        save_results (bool): Whether to save the benchmark results.
        language (str or list of str): The languages to include in the list.
        task (str or list of str): The tasks to consider in the list.
        evaluate_train (bool): Whether to evaluate the training set as well.
        verbose (bool): Whether to output additional output.
        train_size (list of int): The training sizes to use.
        use_auth_token (bool): Whether an authentication token should be used.
        benchmark_results (dict): The benchmark results.
    '''
    def __init__(self,
                 progress_bar: bool = True,
                 save_results: bool = False,
                 language: Union[str, List[str]] = ['da', 'sv', 'no'],
                 model_language: Optional[Union[str, List[str]]] = None,
                 dataset_language: Optional[Union[str, List[str]]] = None,
                 task: Union[str, List[str]] = 'all',
                 evaluate_train: bool = False,
                 train_size: List[int] = [1024],
                 raise_error_on_invalid_model: bool = False,
                 use_auth_token: bool = False,
                 verbose: bool = False):

        # If `language` contains 'no' then also include 'nb' and 'nn'. Conversely, if
        # both 'nb' and 'nn' are specified then only include 'no'.
        if 'no' in language or ('nb' in language and 'nn' in language):
            language = list(set(language) | {'nb', 'nn', 'no'})

        # Set parameters
        self.progress_bar = progress_bar
        self.save_results = save_results
        self.language = language
        self.model_language = model_language
        self.dataset_language = dataset_language
        self.task = task
        self.evaluate_train = evaluate_train
        self.train_size = train_size
        self.raise_error_on_invalid_model = raise_error_on_invalid_model
        self.use_auth_token = use_auth_token
        self.verbose = verbose

        # Initialise variable storing model lists, so we only have to fetch it
        # once
        self._model_lists = None

        # Initialise variable storing all benchmark results, which will be
        # updated as more models are benchmarked
        self.benchmark_results = defaultdict(dict)

        # Set logging level based on verbosity
        logging_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(logging_level)

        # Update the list of benchmarks
        logger.info('Updating the list of benchmark datasets')
        self._update_benchmarks(evaluate_train=evaluate_train,
                                use_auth_token=use_auth_token,
                                verbose=verbose)

    def _update_benchmarks(self, **params):
        '''Updates the internal list of all benchmarks.

        This list will be stored in the `_benchmarks` variable.

        Args:
            params:
                Dictionary of benchmark parameters.
        '''
        self._benchmarks = [(short_name, name, cls(**params))
                            for short_name, name, cls, _ in get_all_datasets()]

    @staticmethod
    def _get_model_ids(language: Optional[str] = None,
                       task: Optional[str] = None) -> List[str]:
        '''Retrieves all the model IDs in a given language with a given task.

        Args:
            language (str or None):
                The language code of the language to consider. If None then the
                models will not be filtered on language. Defaults to None.
            task (str or None):
                The task to consider. If None then the models will not be
                filtered on task. Defaults to None.

        Returns:
            list of str: The model IDs of the relevant models.
        '''
        # Set GET request parameter values
        params = dict()
        if language is not None:
            params['language'] = language
        if task is not None:
            params['pipeline_tag'] = task

        # Fetch and parse the html from the HuggingFace Hub
        url = 'https://huggingface.co/models'
        html = requests.get(url, params=params).text
        soup = BeautifulSoup(html, 'html.parser')

        # Extract the model ids from the html
        articles = soup.find_all('article')
        model_ids = [header['title']
                     for article in articles
                     for header in article.find_all('header')
                     if header.get('class') is not None and
                     header.get('title') is not None and
                     'items-center' in header['class']]

        return model_ids

    def _get_model_lists(self,
                         languages: List[str],
                         tasks: List[str]) -> Dict[str, List[str]]:
        '''Fetches up-to-date model lists.

        Args:
            languages (list of either str or None):
                The language codes of the language to consider. If None is
                present in the list then the models will not be filtered on
                language.
            tasks (list of either str or None):
                The task to consider. If None is present in the list then the
                models will not be filtered on task.

        Returns:
            dict:
                The keys are filterings of the list, which includes all
                language codes, including 'multilingual', all tasks, as well as
                'all'. The values are lists of model IDs.
        '''
        # Log fetching message
        log_msg = 'Fetching list of models'
        if None not in languages:
            log_msg += f' for the languages {languages}'
            if None not in tasks:
                log_msg += f' and tasks {tasks}'
        else:
            if None not in tasks:
                log_msg += f' for the tasks {tasks}'
        log_msg += ' from the HuggingFace Hub.'
        logger.info(log_msg)

        # Initialise model lists
        model_lists = defaultdict(list)
        for language in languages:
            for task in tasks:
                model_ids = self._get_model_ids(language, task)
                model_lists['all'].extend(model_ids)
                model_lists[language].extend(model_ids)
                model_lists[task].extend(model_ids)

        # Add multilingual models manually
        multi_models = ['xlm-roberta-base',
                        'xlm-roberta-large',
                        'bert-base-multilingual-cased',
                        'distilbert-base-multilingual-cased',
                        'cardiffnlp/twitter-xlm-roberta-base']
        model_lists['multilingual'] = multi_models
        model_lists['all'].extend(multi_models)

        # Add random models
        random_models = ['random-roberta-sequence-clf',
                         'random-roberta-token-clf']
        model_lists['all'].extend(random_models)

        # Add some multilingual Danish models manually that have not marked
        # 'da' as their language
        if 'da' in languages:
            multi_da_models = ['Geotrend/bert-base-en-da-cased',
                               'Geotrend/bert-base-25lang-cased',
                               'Geotrend/bert-base-en-fr-de-no-da-cased',
                               'Geotrend/distilbert-base-en-da-cased',
                               'Geotrend/distilbert-base-25lang-cased',
                               'Geotrend/distilbert-base-en-fr-de-no-da-cased']
            model_lists['da'].extend(multi_da_models)
            model_lists['all'].extend(multi_da_models)

        # Add some multilingual Norwegian models manually that have not marked
        # 'no' as their language
        if 'no' in languages:
            multi_no_models = ['Geotrend/bert-base-en-no-cased',
                               'Geotrend/bert-base-25lang-cased',
                               'Geotrend/bert-base-en-fr-de-no-da-cased',
                               'Geotrend/distilbert-base-en-no-cased',
                               'Geotrend/distilbert-base-25lang-cased',
                               'Geotrend/distilbert-base-en-fr-de-no-da-cased']
            model_lists['no'].extend(multi_no_models)
            model_lists['all'].extend(multi_no_models)

        # Remove duplicates from the lists
        for lang, model_list in model_lists.items():
            model_lists[lang] = list(set(model_list))

        return model_lists

    def benchmark(self,
                  model_id: Optional[Union[List[str], str]] = None,
                  dataset: Optional[Union[List[str], str]] = None,
                  progress_bar: Optional[bool] = None,
                  save_results: Optional[bool] = None,
                  language: Optional[Union[str, List[str]]] = None,
                  model_language: Optional[Union[str, List[str]]] = None,
                  dataset_language: Optional[Union[str, List[str]]] = None,
                  task: Optional[Union[str, List[str]]] = None,
                  evaluate_train: Optional[bool] = None,
                  use_auth_token: Optional[bool] = None,
                  verbose: Optional[bool] = None
                  ) -> Dict[str, Dict[str, dict]]:
        '''Benchmarks models on datasets.

        Args:
            model_id (str, list of str or None, optional):
                The model ID(s) of the models to benchmark. If None then all
                relevant model IDs will be benchmarked. Defaults to None.
            dataset (str, list of str or None, optional):
                The datasets to benchmark on. If None then all datasets will
                be benchmarked. Defaults to None.
            progress_bar (bool or None, optional):
                Whether progress bars should be shown. If None then the default
                value from the constructor will be used. Defaults to None.
            save_results (bool or None, optional):
                Whether to save the benchmark results to
                'scandeval_benchmark_results.json'. If None then the default
                value from the constructor will be used. Defaults to None.
            language (str, list of str or None, optional):
                The language codes of the languages to include in the list. Set
                this to 'all' if all languages (also non-Scandinavian) should
                be considered. If None then the default value from the
                constructor will be used. Defaults to None.
            model_language (str, list of str or None, optional):
                The language codes of the languages to include in the list. Set
                this to 'all' if all languages (also non-Scandinavian) should
                be considered. If None then the default value from the
                constructor will be used. Defaults to None.
            dataset_language (str, list of str or None, optional):
                The language codes of the languages to include in the list. Set
                this to 'all' if all languages (also non-Scandinavian) should
                be considered. If None then the default value from the
                constructor will be used. Defaults to None.
            task (str, list of str or None, optional):
                The tasks to consider in the list. Set this to 'all' if all
                tasks should be considered. If None then the default value from
                the constructor will be used. Defaults to None.
            evaluate_train (bool or None, optional):
                Whether to evaluate the training set as well. If None then the
                default value from the constructor will be used. Defaults to
                None.
            use_auth_token (bool, optional):
                Whether the benchmark should use an authentication token. If
                None then the default value from the constructor will be used.
                Defaults to None.
            verbose (bool or None, optional):
                Whether to output additional output. If None then the default
                value from the constructor will be used. Defaults to None.

        Returns:
            dict:
                A nested dictionary of the benchmark results. The keys are the
                names of the datasets, with values being new dictionaries
                having the model IDs as keys.
        '''
        # Set default values if the arguments are not set
        if progress_bar is None:
            progress_bar = self.progress_bar
        if save_results is None:
            save_results = self.save_results
        if language is None:
            language = self.language
        if model_language is None:
            model_language = self.model_language
        if dataset_language is None:
            dataset_language = self.dataset_language
        if task is None:
            task = self.task
        if evaluate_train is None:
            evaluate_train = self.evaluate_train
        if use_auth_token is None:
            use_auth_token = self.use_auth_token
        if verbose is None:
            verbose = self.verbose

        # Update benchmark list
        if (evaluate_train != self.evaluate_train or verbose != self.verbose):
            logger.info('Updating the list of benchmark datasets')
            self._update_benchmarks(evaluate_train=evaluate_train,
                                    verbose=verbose)

        # Ensure that `language` is a list
        if language == 'all':
            languages = [None]
        elif isinstance(language, str):
            languages = [language]
        else:
            languages = language

        # Ensure that `model_language` is a list
        if model_language == 'all':
            model_languages = [None]
        elif isinstance(model_language, str):
            model_languages = [model_language]
        elif model_language is None:
            model_languages = languages
        else:
            model_languages = model_language

        # Ensure that `dataset_language` is a list
        if dataset_language == 'all':
            dataset_languages = [None]
        elif isinstance(dataset_language, str):
            dataset_languages = [dataset_language]
        elif dataset_language is None:
            dataset_languages = languages
        else:
            dataset_languages = dataset_language

        # Ensure that `task` is a list
        if task == 'all':
            tasks = [None]
        elif isinstance(task, str):
            tasks = [task]
        else:
            tasks = task

        # If `model_id` is not specified, then fetch all the relevant model IDs
        if model_id is None:

            # If the model lists have not been fetched already, then do it
            if self._model_lists is None:
                self._model_lists = self._get_model_lists(languages=model_languages,
                                                          tasks=tasks)
            try:
                model_ids = list()
                for language in model_languages:
                    model_ids.extend(self._model_lists[language])
                for task in tasks:
                    model_ids.extend(self._model_lists[task])
                model_ids.extend(self._model_lists['multilingual'])

            # If the model list corresponding to the language or task was not
            # present in the stored model lists, then fetch new model lists and
            # try again
            except KeyError:
                self._model_lists = self._get_model_lists(languages=model_languages,
                                                          tasks=tasks)
                model_ids = list()
                for language in model_languages:
                    model_ids.extend(self._model_lists[language])
                for task in tasks:
                    model_ids.extend(self._model_lists[task])
                model_ids.extend(self._model_lists['multilingual'])

            # Remove duplicate model IDs
            model_ids = list(set(model_ids))

        # Define `model_ids` variable, storing all the relevant model IDs
        elif isinstance(model_id, str):
            model_ids = [model_id]
        else:
            model_ids = model_id

        # Define `datasets` variable, storing all the relevant datasets
        if dataset is None:
            datasets = [
                d for d, _, obj in self._benchmarks
                if obj.language in dataset_languages
            ]
        elif isinstance(dataset, str):
            datasets = [dataset]
        else:
            datasets = dataset

        # Fetch the benchmark datasets, filtered by the `datasets` variable
        benchmarks = [
            (dataset, alias, obj)
            for dataset, alias, obj in self._benchmarks
            if dataset in datasets
        ]

        # Benchmark all the models in `model_ids` on all the datasets in
        # `benchmarks`
        for dataset, alias, obj in benchmarks:
            for train_size in self.train_size:
                for model_id in model_ids:
                    logger.info(f'Benchmarking {model_id} on {alias} with '
                                f'{train_size} samples:')
                    try:
                        params = dict(progress_bar=progress_bar,
                                      train_size=train_size,
                                      use_auth_token=use_auth_token)
                        results = obj(model_id, **params)
                        self.benchmark_results[dataset][model_id] = results
                        logger.debug(f'Results:\n{results}')

                    except InvalidBenchmark as e:

                        # If the model ID is not valid then raise an error, if
                        # specified
                        model_err_msg = 'could not be loaded from the Hugging Face Hub'
                        if (self.raise_error_on_invalid_model and
                                str(e).endswith(model_err_msg)):
                            raise e

                        # Otherwise, log the error
                        else:
                            logger.info(f'{model_id} could not be benchmarked '
                                        f'on {alias}. Skipping.')
                            logger.debug(f'The error message was "{e}".')

        # Save the benchmark results
        if save_results:
            output_path = Path.cwd() / 'scandeval_benchmark_results.json'
            with output_path.open('w') as f:
                json.dump(self.benchmark_results, f)

        return self.benchmark_results

    def __call__(self, *args, **kwargs):
        return self.benchmark(*args, **kwargs)
