'''Functions that load datasets'''

import requests
import json
from typing import Tuple, Union, List
from .utils import get_all_datasets


def load_dataset(name: str) -> Tuple[dict, dict, dict, dict]:
    '''Load a benchmark dataset.

    Args:
        name (str):
            Name of the dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.

    Raises:
        RuntimeError:
            If `name` is not a valid dataset name.
    '''
    dataset_names = [name for name, _, _, _ in get_all_datasets()]
    if name in dataset_names:
        loader = [loader for dataset_name, _, _, loader in get_all_datasets()
                  if dataset_name == name][0]
        return loader()
    else:
        raise RuntimeError(f'The dataset "{name}" was not recognised.')


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


def load_norne_nb() -> Tuple[dict, dict, dict, dict]:
    '''Load the the Bokmål part of the NorNE dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/norne_nb/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'ner_tags')
    return X_train, X_test, y_train, y_test


def load_norne_nn() -> Tuple[dict, dict, dict, dict]:
    '''Load the the Nynorsk part of the NorNE dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/norne_nn/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'ner_tags')
    return X_train, X_test, y_train, y_test


def load_nordial() -> Tuple[dict, dict, dict, dict]:
    '''Load the Bokmål/Nynorsk part of the NorDial dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/nordial/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_norec() -> Tuple[dict, dict, dict, dict]:
    '''Load the NoReC dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/norec/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


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


def load_dane_no_misc() -> Tuple[dict, dict, dict, dict]:
    '''Load the the DaNE dataset without MISC tags.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dane_no_misc/')
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


def load_twitter_subj() -> Tuple[dict, dict, dict, dict]:
    '''Load the TwitterSubj dataset.

    This dataset is the subjectivity part of the TwitterSent dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/twitter_subj/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'tweet', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'tweet', 'label')
    return X_train, X_test, y_train, y_test


def load_europarl_subj() -> Tuple[dict, dict, dict, dict]:
    '''Load the EuroparlSubj dataset.

    This dataset is the subjectivity part of the Europarl2 dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/europarl_subj/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_europarl_sent() -> Tuple[dict, dict, dict, dict]:
    '''Load the EuroparlSent dataset.

    This dataset is the sentiment part of the Europarl2 dataset.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/europarl_sent/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_lcc() -> Tuple[dict, dict, dict, dict]:
    '''Load the LCC dataset.

    This dataset is the concatenation of the LCC1 and LCC2 datasets.

    Returns:
        tuple:
            Four dicts, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/lcc/')
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
