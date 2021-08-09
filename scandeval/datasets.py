'''Functions that load datasets'''

import requests
import json
from typing import Tuple, Union, List


def _get_dataset_from_url(url: str,
                          feature_key: Union[str, List[str]],
                          label_key: Union[str, List[str]]
                          ) -> Tuple[dict, dict]:
    response = requests.get(url)
    records = response.text.split('\n')
    data = [json.loads(record) for record in records if record != '']

    if isinstance(feature_key, str):
        feature_key = [feature_key]
    if isinstance(label_key, str):
        label_key = [label_key]

    docs = {key: [data_dict[key] for data_dict in data] for key in feature_key}
    labels = {key: [data_dict[key] for data_dict in data] for key in label_key}

    return docs, labels


def load_angry_tweets() -> Tuple[dict, dict, dict, dict]:
    '''Load the AngryTweets dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_dane() -> Tuple[dict, dict, dict, dict]:
    '''Load the the DaNE dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dane/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'ner_tags')
    return X_train, X_test, y_train, y_test


def load_ddt_pos() -> Tuple[dict, dict, dict, dict]:
    '''Load the POS part of the DDT dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dane/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'pos_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'pos_tags')
    return X_train, X_test, y_train, y_test


def load_ddt_dep() -> Tuple[dict, dict, dict, dict]:
    '''Load the dependency parsing part of the DDT dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dane/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             ['heads', 'deps'])
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           ['heads', 'deps'])
    return X_train, X_test, y_train, y_test


def load_dkhate() -> Tuple[dict, dict, dict, dict]:
    '''Load the DKHate dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_europarl1() -> Tuple[dict, dict, dict, dict]:
    '''Load the Europarl1 dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_europarl2() -> Tuple[dict, dict, dict, dict]:
    '''Load the Europarl2 dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_lcc1() -> Tuple[dict, dict, dict, dict]:
    '''Load the LCC1 dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_lcc2() -> Tuple[dict, dict, dict, dict]:
    '''Load the LCC2 dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_twitter_sent() -> Tuple[dict, dict, dict, dict]:
    '''Load the TwitterSent dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
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
