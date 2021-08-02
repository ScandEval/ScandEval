'''Fetches an updated list of all Scandinavian models on the HuggingFace Hub'''

import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Union
from collections import defaultdict
import logging

from .dane import DaneBenchmark


class Benchmark:
    '''Benchmarking all the Scandinavian language models.

    Args
        languages (list of str, optional):
            The language codes of the languages to include in the list.
            Defaults to ['da', 'sv', 'no'].
        tasks (list of str, optional):
            The tasks to consider in the list. Defaults to ['fill-mask',
            'token-classification', 'text-classification'].
    '''
    def __init__(self,
                 languages: List[str] = ['da', 'sv', 'no'],
                 tasks = ['fill-mask',
                          'token-classification',
                          'text-classification'],
                 verbose: bool = False):
        self.languages = languages
        self.tasks = tasks
        self._model_list = self._get_model_list()
        self.benchmark_results = defaultdict(dict)
        params = dict(verbose=verbose)
        self._evaluators = {
            'dane': DaneBenchmark(**params),
            'dane_no_misc': DaneBenchmark(include_misc_tags=False, **params)
        }

        if verbose:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        format = '%(asctime)s [%(levelname)s] <%(name)s> %(message)s'
        logging.basicConfig(level=logging_level, format=format)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_model_ids(language: str, task: str) -> List[str]:
        '''Retrieves all the model IDs in a given language with a given task.

        Args:
            language (str): The language code of the language to consider.
            task (str): The task to consider.

        Returns:
            list of str: The model IDs of the relevant models.
        '''
        url = 'https://huggingface.co/models'
        params = dict(filter=language, pipeline_tag=task)
        html = requests.get(url, params=params).text
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('article')
        model_ids = [header['title'] for article in articles
                                     for header in article.find_all('header')
                                     if header.get('class') is not None and
                                        header.get('title') is not None and
                                        'items-center' in header['class']]
        return model_ids

    def _get_model_list(self):
        '''Updates the model list'''
        # Get new model list
        new_model_list = list()
        for language in self.languages:
            for task in self.tasks:
                model_ids = self._get_model_ids(language, task)
                new_model_list.extend(model_ids)

        # Add XLM-RoBERTa models manually
        new_model_list.extend(['xlm-roberta-base', 'xlm-roberta-large'])

        # Save model list
        return new_model_list

    def __call__(self, model_ids: Optional[Union[List[str], str]] = None):
        '''Benchmarks all models in the model list.

        Args:
            TODO

        Returns:
            TODO
        '''
        if model_ids is None:
            model_ids = self._get_model_list()
        elif isinstance(model_ids, str):
            model_ids = [model_ids]

        for name, evaluator in self._evaluators.items():
            for model_id in model_ids:
                self.logger.info(f'Evaluating {model_id} on {name}:')
                results = evaluator(model_id)
                self.benchmark_results[name][model_id] = results

        return self.benchmark_results
