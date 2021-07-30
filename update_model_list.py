'''Fetches an updated list of all Scandinavian models on the HuggingFace Hub'''

import requests
from bs4 import BeautifulSoup
import yaml
from pathlib import Path
from typing import List
import logging


logger = logging.getLogger(__name__)


LANGUAGES = ['da', 'se', 'no']
TASKS = ['fill-mask', 'token-classification', 'text-classification']


def get_model_ids(language: str, task: str):
    '''Retrieves all the model IDs in a given language with a given task.

    Args:
        TODO

    Returns:
        TODO
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


def update_model_list(languages: List[str] = ['da', 'se', 'no'],
                      tasks: List[str] = ['fill-mask',
                                          'token-classification',
                                          'text-classification']):
    '''Updates the model list.

    Args
        TODO
    '''
    # Initialise path to the model list file
    yaml_path = Path('data') / 'model_list.yaml'

    # Get new model list
    new_model_list = list()
    for language in languages:
        for task in tasks:
            model_ids = get_model_ids(language, task)
            new_model_list.extend(model_ids)

    # Read current model list
    with yaml_path.open('r') as f:
        old_model_list = yaml.safe_load(f)
    if old_model_list is None:
        old_model_list = list()

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

    with yaml_path.open('w') as f:
        yaml.safe_dump(new_model_list, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    update_model_list()
