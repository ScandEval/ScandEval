'''Functions that load datasets'''

import requests
import json
from typing import Tuple, Union, List
import pandas as pd
from .utils import get_all_datasets


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, pd.DataFrame]:
    '''Load a benchmark dataset.

    Args:
        name (str):
            Name of the dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    response = requests.get(url)
    records = response.text.split('\n')
    data = [json.loads(record) for record in records if record != '']

    if isinstance(feature_key, str):
        feature_key = [feature_key]
    if isinstance(label_key, str):
        label_key = [label_key]

    docs = {key: [data_dict[key] for data_dict in data] for key in feature_key}
    labels = {key: [data_dict[key] for data_dict in data] for key in label_key}

    return pd.DataFrame(docs), pd.DataFrame(labels)


def load_suc3() -> Tuple[pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame]:
    '''Load the SUC 3.0 dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/suc3/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'ner_tags')
    return X_train, X_test, y_train, y_test


def load_norne_nb() -> Tuple[pd.DataFrame, pd.DataFrame,
                             pd.DataFrame, pd.DataFrame]:
    '''Load the the Bokmål part of the NorNE dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_norne_nn() -> Tuple[pd.DataFrame, pd.DataFrame,
                             pd.DataFrame, pd.DataFrame]:
    '''Load the the Nynorsk part of the NorNE dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_nordial() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the Bokmål/Nynorsk part of the NorDial dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_norec() -> Tuple[pd.DataFrame, pd.DataFrame,
                          pd.DataFrame, pd.DataFrame]:
    '''Load the NoReC dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_norec_is() -> Tuple[pd.DataFrame, pd.DataFrame,
                             pd.DataFrame, pd.DataFrame]:
    '''Load the NoReC-IS dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/norec_is/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_norec_fo() -> Tuple[pd.DataFrame, pd.DataFrame,
                             pd.DataFrame, pd.DataFrame]:
    '''Load the NoReC-FO dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/norec_fo/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_angry_tweets() -> Tuple[pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame, pd.DataFrame]:
    '''Load the AngryTweets dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/angry_tweets/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_dane() -> Tuple[pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, pd.DataFrame]:
    '''Load the the DaNE dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_ndt_nb_pos() -> Tuple[pd.DataFrame, pd.DataFrame,
                               pd.DataFrame, pd.DataFrame]:
    '''Load the Bokmål POS part of the NDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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
                                             'pos_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'pos_tags')
    return X_train, X_test, y_train, y_test


def load_ndt_nn_pos() -> Tuple[pd.DataFrame, pd.DataFrame,
                               pd.DataFrame, pd.DataFrame]:
    '''Load the Nynorsk POS part of the NDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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
                                             'pos_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'pos_tags')
    return X_train, X_test, y_train, y_test


def load_ndt_nb_dep() -> Tuple[pd.DataFrame, pd.DataFrame,
                               pd.DataFrame, pd.DataFrame]:
    '''Load the Bokmål POS part of the NDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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
                                             ['heads', 'deps'])
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           ['heads', 'deps'])
    return X_train, X_test, y_train, y_test


def load_ndt_nn_dep() -> Tuple[pd.DataFrame, pd.DataFrame,
                               pd.DataFrame, pd.DataFrame]:
    '''Load the Nynorsk POS part of the NDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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
                                             ['heads', 'deps'])
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           ['heads', 'deps'])
    return X_train, X_test, y_train, y_test


def load_ddt_pos() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the POS part of the DDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_ddt_dep() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the dependency parsing part of the DDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_dkhate() -> Tuple[pd.DataFrame, pd.DataFrame,
                           pd.DataFrame, pd.DataFrame]:
    '''Load the DKHate dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dkhate/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_europarl() -> Tuple[pd.DataFrame, pd.DataFrame,
                             pd.DataFrame, pd.DataFrame]:
    '''Load the Europarl dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/europarl/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_lcc() -> Tuple[pd.DataFrame, pd.DataFrame,
                        pd.DataFrame, pd.DataFrame]:
    '''Load the LCC dataset.

    This dataset is the concatenation of the LCC1 and LCC2 datasets.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
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


def load_twitter_sent() -> Tuple[pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame, pd.DataFrame]:
    '''Load the TwitterSent dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/twitter_sent/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_dalaj() -> Tuple[pd.DataFrame, pd.DataFrame,
                          pd.DataFrame, pd.DataFrame]:
    '''Load the DaLaJ dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/dalaj/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_absabank_imm() -> Tuple[pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame, pd.DataFrame]:
    '''Load the ABSAbank-Imm dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/absabank_imm/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url, 'text', 'label')
    X_test, y_test = _get_dataset_from_url(test_url, 'text', 'label')
    return X_train, X_test, y_train, y_test


def load_sdt_pos() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the POS part of the SDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/sdt/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'pos_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'pos_tags')
    return X_train, X_test, y_train, y_test


def load_sdt_dep() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the dependency parsing part of the SDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/sdt/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             ['heads', 'deps'])
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           ['heads', 'deps'])
    return X_train, X_test, y_train, y_test


def load_idt_pos() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the POS part of the IDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/idt/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'pos_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'pos_tags')
    return X_train, X_test, y_train, y_test


def load_idt_dep() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the dependency parsing part of the IDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/idt/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             ['heads', 'deps'])
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           ['heads', 'deps'])
    return X_train, X_test, y_train, y_test


def load_mim_gold_ner() -> Tuple[pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame, pd.DataFrame]:
    '''Load the the MIM-GOLD-NER dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/mim_gold_ner/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'ner_tags')
    return X_train, X_test, y_train, y_test


def load_wikiann_fo() -> Tuple[pd.DataFrame, pd.DataFrame,
                               pd.DataFrame, pd.DataFrame]:
    '''Load the the Faroese WikiANN dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/wikiann_fo/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'ner_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'ner_tags')
    return X_train, X_test, y_train, y_test


def load_fdt_pos() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the POS part of the FDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/fdt/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             'pos_tags')
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           'pos_tags')
    return X_train, X_test, y_train, y_test


def load_fdt_dep() -> Tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    '''Load the dependency parsing part of the FDT dataset.

    Returns:
        tuple:
            Four dataframes, `X_train`, `X_test`, `y_train` and `y_test`, where
            `X_train` and `X_test` corresponds to the feature matrices for the
            training and test split, respectively, and `y_train` and `y_test`
            contains the target vectors.
    '''
    base_url = ('https://raw.githubusercontent.com/saattrupdan/ScandEval/'
                'main/datasets/fdt/')
    train_url = base_url + 'train.jsonl'
    test_url = base_url + 'test.jsonl'
    X_train, y_train = _get_dataset_from_url(train_url,
                                             ['doc', 'tokens'],
                                             ['heads', 'deps'])
    X_test, y_test = _get_dataset_from_url(test_url,
                                           ['doc', 'tokens'],
                                           ['heads', 'deps'])
    return X_train, X_test, y_train, y_test
