'''Fetches an updated list of all Scandinavian models on the HuggingFace Hub'''

import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Union, Dict
from collections import defaultdict
import logging
import json
from pathlib import Path

from .dane import DaneBenchmark
from .ddt_pos import DdtPosBenchmark
from .ddt_dep import DdtDepBenchmark
from .angry_tweets import AngryTweetsBenchmark
from .twitter_sent import TwitterSentBenchmark
from .europarl1 import Europarl1Benchmark
from .europarl2 import Europarl2Benchmark
from .lcc1 import Lcc1Benchmark
from .lcc2 import Lcc2Benchmark
from .dkhate import DkHateBenchmark
from .utils import InvalidBenchmark


logger = logging.getLogger(__name__)


class Benchmark:
    '''Benchmarking all the Scandinavian language models.

    Args
        language (str or list of str, optional):
            The language codes of the languages to include in the list.
            Defaults to ['da', 'sv', 'no'].
        task (str or list of str, optional):
            The tasks to consider in the list. Defaults to ['fill-mask',
            'token-classification', 'text-classification'].
    '''
    def __init__(self,
                 language: Union[str, List[str]] = ['da', 'sv', 'no'],
                 task: Union[str, List[str]] = ['fill-mask',
                                                'token-classification',
                                                'text-classification'],
                 batch_size: int = 32,
                 verbose: bool = False):
        self.languages = [language] if isinstance(language, str) else language
        self.tasks = [task] if isinstance(task, str) else task
        self._model_lists = self._get_model_lists()
        self.benchmark_results = defaultdict(dict)
        params = dict(verbose=verbose, batch_size=batch_size)
        self._benchmarks = [
            ('dane', 'DaNE with MISC tags', DaneBenchmark(**params)),
            ('dane-no-misc', 'DaNE without MISC tags',
             DaneBenchmark(include_misc_tags=False, **params)),
            ('ddt-pos', 'the POS part of DDT', DdtPosBenchmark(**params)),
            ('ddt-dep', 'the DEP part of DDT', DdtDepBenchmark(**params)),
            ('angry-tweets', 'Angry Tweets', AngryTweetsBenchmark(**params)),
            ('twitter-sent', 'Twitter Sent', TwitterSentBenchmark(**params)),
            ('dkhate', 'DKHate', DkHateBenchmark(**params)),
            ('europarl1', 'Europarl1', Europarl1Benchmark(**params)),
            ('europarl2', 'Europarl2', Europarl2Benchmark(**params)),
            ('lcc1', 'LCC1', Lcc1Benchmark(**params)),
            ('lcc2', 'LCC2', Lcc2Benchmark(**params))
        ]

        if verbose:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        logger.setLevel(logging_level)

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
        model_ids = [header['title']
                     for article in articles
                     for header in article.find_all('header')
                     if header.get('class') is not None and
                     header.get('title') is not None and
                     'items-center' in header['class']]
        return model_ids

    def _get_model_lists(self) -> Dict[str, List[str]]:
        '''Updates the model list'''

        # Initialise model lists
        model_lists = defaultdict(list)
        for language in self.languages:
            for task in self.tasks:
                model_ids = self._get_model_ids(language, task)
                model_lists['all'].extend(model_ids)
                model_lists[language].extend(model_ids)
                model_lists[task].extend(model_ids)

        # Add multilingual models manually
        multilingual_models = ['xlm-roberta-base', 'xlm-roberta-large']
        model_lists['all'].extend(multilingual_models)
        model_lists['multilingual'] = multilingual_models

        # Save model list
        return model_lists

    def __call__(self,
                 model_id: Optional[Union[List[str], str]] = None,
                 dataset: Optional[Union[List[str], str]] = None,
                 num_finetunings: int = 10,
                 save_results: bool = False
                 ) -> Dict[str, Dict[str, dict]]:
        '''Benchmarks all models in the model list.

        Args:
            model_id (str, list of str or None, optional):
                The model ID(s) of the models to benchmark. If None then all
                relevant model IDs will be benchmarked. Defaults to None.
            dataset (str, list of str or None, optional):
                The datasets to benchmark on. If None then all datasets will
                be benchmarked. Defaults to None.
            num_finetunings (int, optional):
                The number of times to finetune each model on. Defaults to 10.
            save_results (bool, optional):
                Whether to save the benchmark results to
                'scandeval_benchmark_results.json'. Defaults to False.

        Returns:
            dict:
                A nested dictionary of the benchmark results. The keys are the
                names of the datasets, with values being new dictionaries
                having the model IDs as keys.
        '''
        if model_id is None:
            model_ids = self._model_lists['all']
        elif isinstance(model_id, str):
            model_ids = [model_id]
        else:
            model_ids = model_id

        if dataset is None:
            datasets = [d for d, _, _ in self._benchmarks]
        elif isinstance(dataset, str):
            datasets = [dataset]
        else:
            datasets = dataset

        benchmarks = [(dataset, alias, cls)
                      for dataset, alias, cls in self._benchmarks
                      if dataset in datasets]

        for dataset, alias, cls in benchmarks:
            for model_id in model_ids:
                logger.info(f'Benchmarking {model_id} on {alias}:')
                try:
                    results = cls(model_id, num_finetunings=num_finetunings)
                    self.benchmark_results[dataset][model_id] = results
                    logger.debug(f'Results:\n{results}')
                except InvalidBenchmark as e:
                    logger.info(f'{model_id} could not be benchmarked '
                                f'on {alias}. Skipping.')
                    logger.debug(f'The error message was {e}.')

        # Save the benchmark results
        if save_results:
            output_path = Path.cwd() / 'scandeval_benchmark_results.json'
            with output_path.open('w') as f:
                json.dump(self.benchmark_results, f)

        return self.benchmark_results
