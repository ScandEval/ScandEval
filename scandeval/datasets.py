'''Functions that load datasets'''

import requests
import json
from typing import Tuple


def _get_dataset_from_url(url: str,
                          feature_key: str,
                          label_key: str) -> Tuple[list, list]:
    response = requests.get(url)
    records = response.text.split('\n')
    data = [json.loads(record) for record in records if record != '']
    docs = [data_dict[feature_key] for data_dict in data]
    labels = [data_dict[label_key] for data_dict in data]
    return docs, labels


def load_angry_tweets() -> Tuple[list, list, list, list]:
    '''Load the AngryTweets dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/angry_tweets/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'tweet', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'tweet', 'label')
    return X_train, X_test, y_train, y_test


def load_dane() -> Tuple[list, list, list, list]:
    '''Load the DaNE dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dane/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'tokens', 'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url, 'tokens', 'ner_tags')
    return X_train, X_test, y_train, y_test


def load_dkhate() -> Tuple[list, list, list, list]:
    '''Load the DKHate dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dkhate/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'tweet', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'tweet', 'label')
    return X_train, X_test, y_train, y_test


def load_europarl1() -> Tuple[list, list, list, list]:
    '''Load the Europarl1 dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/europarl1/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_europarl2() -> Tuple[list, list, list, list]:
    '''Load the Europarl2 dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/europarl2/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_lcc1() -> Tuple[list, list, list, list]:
    '''Load the LCC1 dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/lcc1/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_lcc2() -> Tuple[list, list, list, list]:
    '''Load the LCC2 dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/lcc2/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_twitter_sent() -> Tuple[list, list, list, list]:
    '''Load the TwitterSent dataset.

    Returns:
        tuple:
            Four lists, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/twitter_sent/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'tweet', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'tweet', 'label')
    return X_train, X_test, y_train, y_test
