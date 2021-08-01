'''Fetches an updated list of all Scandinavian models on the HuggingFace Hub'''

import requests
from bs4 import BeautifulSoup
import yaml
from pathlib import Path
from typing import List, Optional, Union
from collections import defaultdict
import logging
from scandeval.dane import DaneEvaluator


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(name)s> %(message)s')
logger = logging.getLogger(__name__)


class Benchmarker:
    '''Benchmarking all the Scandinavian language models.

    Args
        path (str, optional):
            The path to the model list. Defaults to 'data/model_list.yaml'.
        languages (list of str, optional):
            The language codes of the languages to include in the list.
            Defaults to ['da', 'sv', 'no'].
        tasks (list of str, optional):
            The tasks to consider in the list. Defaults to ['fill-mask',
            'token-classification', 'text-classification'].
    '''
    def __init__(self,
                 path: str = 'data/model_list.yaml',
                 languages: List[str] = ['da', 'sv', 'no'],
                 tasks = ['fill-mask',
                          'token-classification',
                          'text-classification']):
        self.path = Path(path)
        self.languages = languages
        self.tasks = tasks
        self._model_list = self._get_model_list()
        self.benchmark_results = defaultdict(dict)
        self._evaluators = {
            'dane': DaneEvaluator(),
            'dane_no_misc': DaneEvaluator(include_misc_tags=False)
        }

    def _get_model_list(self) -> List[str]:
        '''Get the current model list.

        Returns:
            list of str: The model list.
        '''
        with self.path.open('r') as f:
            model_list = yaml.safe_load(f)
        if model_list is None:
            model_list = list()
        return model_list

    def _set_model_list(self, new_model_list: List[str]):
        '''Update the model list.

        Args:
            list of str: New model list.
        '''
        self.model_list = new_model_list
        with self.path.open('w') as f:
            yaml.safe_dump(new_model_list, f)

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

    def _update_model_list(self):
        '''Updates the model list'''
        # Get new model list
        new_model_list = list()
        for language in self.languages:
            for task in self.tasks:
                model_ids = self._get_model_ids(language, task)
                new_model_list.extend(model_ids)

        # Add XLM-RoBERTa models manually
        new_model_list.extend(['xlm-roberta-base', 'xlm-roberta-large'])

        # Read current model list
        old_model_list = self._get_model_list()

        # Log what models were added
        models_added = [model for model in new_model_list
                              if model not in old_model_list]
        if len(models_added) > 0:
            logger.info(f'Added the following new models: {models_added}')

        # Log what models were removed
        models_removed = [model for model in old_model_list
                                if model not in new_model_list]
        if len(models_removed) > 0:
            logger.info(f'Removed the following models: {models_removed}')

        if len(models_added) == 0 and len(models_removed) == 0:
            logger.info('No changes in model list.')

        self._set_model_list(new_model_list)

    def benchmark(self, model_ids: Optional[Union[List[str], str]] = None):
        '''Benchmarks all models in the model list.

        Args:
            TODO

        Returns:
            TODO
        '''

        if model_ids is None:
            self._update_model_list()
            model_ids = self.model_list
        elif isinstance(model_ids, str):
            model_ids = [model_ids]

        for name, evaluator in self._evaluators.items():
            for model_id in model_ids:
                results = evaluator(model_id)
                self.benchmark_results[name][model_id] = results

        return self.benchmark_results


if __name__ == '__main__':
    benchmarker = Benchmarker()
    print(benchmarker.benchmark())
