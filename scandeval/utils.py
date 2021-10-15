'''Utility functions to be used in other scripts'''

from functools import wraps
from typing import Callable
import warnings
import datasets.utils.logging as ds_logging
import logging
import pkg_resources
import re
import transformers.utils.logging as tf_logging
import torch
from pydoc import locate
from transformers import (AutoModelForTokenClassification,
                          AutoModelForSequenceClassification,
                          TFAutoModelForTokenClassification,
                          TFAutoModelForSequenceClassification,
                          FlaxAutoModelForTokenClassification,
                          FlaxAutoModelForSequenceClassification,
                          Trainer)


def get_all_datasets() -> list:
    '''Load a list of all datasets.

    Returns:
        list of tuples:
            First entry in each tuple is the short name of the dataset, second
            entry the long name, third entry the benchmark class and fourth
            entry the loading function.
    '''
    return [
        ('dane', 'DaNE',
            locate('scandeval.benchmarks.DaneBenchmark'),
            locate('scandeval.datasets.load_dane')),
        ('ddt-pos', 'the POS part of DDT',
            locate('scandeval.benchmarks.DdtPosBenchmark'),
            locate('scandeval.datasets.load_ddt_pos')),
        ('ddt-dep', 'the DEP part of DDT',
            locate('scandeval.benchmarks.DdtDepBenchmark'),
            locate('scandeval.datasets.load_ddt_dep')),
        ('angry-tweets', 'AngryTweets',
            locate('scandeval.benchmarks.AngryTweetsBenchmark'),
            locate('scandeval.datasets.load_angry_tweets')),
        ('twitter-sent', 'TwitterSent',
            locate('scandeval.benchmarks.TwitterSentBenchmark'),
            locate('scandeval.datasets.load_twitter_sent')),
        ('europarl', 'Europarl',
            locate('scandeval.benchmarks.EuroparlBenchmark'),
            locate('scandeval.datasets.load_europarl')),
        ('dkhate', 'DKHate',
            locate('scandeval.benchmarks.DkHateBenchmark'),
            locate('scandeval.datasets.load_dkhate')),
        ('lcc', 'LCC',
            locate('scandeval.benchmarks.LccBenchmark'),
            locate('scandeval.datasets.load_lcc')),
        ('norec', 'NoReC',
            locate('scandeval.benchmarks.NorecBenchmark'),
            locate('scandeval.datasets.load_norec')),
        ('nordial', 'NorDial',
            locate('scandeval.benchmarks.NorDialBenchmark'),
            locate('scandeval.datasets.load_nordial')),
        ('norne-nb', 'the Bokmål part of NorNE',
            locate('scandeval.benchmarks.NorneNBBenchmark'),
            locate('scandeval.datasets.load_norne_nb')),
        ('norne-nn', 'the Nynorsk part of NorNE',
            locate('scandeval.benchmarks.NorneNNBenchmark'),
            locate('scandeval.datasets.load_norne_nn')),
        ('ndt-nb-pos', 'the Bokmål POS part of NDT',
            locate('scandeval.benchmarks.NdtNBPosBenchmark'),
            locate('scandeval.datasets.load_ndt_nb_pos')),
        ('ndt-nn-pos', 'the Nynorsk POS part of NDT',
            locate('scandeval.benchmarks.NdtNNPosBenchmark'),
            locate('scandeval.datasets.load_ndt_nn_pos')),
        ('ndt-nb-dep', 'the Bokmål DEP part of NDT',
            locate('scandeval.benchmarks.NdtNBDepBenchmark'),
            locate('scandeval.datasets.load_ndt_nb_dep')),
        ('ndt-nn-dep', 'the Nynorsk DEP part of NDT',
            locate('scandeval.benchmarks.NdtNNDepBenchmark'),
            locate('scandeval.datasets.load_ndt_nn_dep')),
        ('dalaj', 'DaLaJ',
            locate('scandeval.benchmarks.DalajBenchmark'),
            locate('scandeval.datasets.load_dalaj')),
        ('absabank-imm', 'ABSAbank-Imm',
            locate('scandeval.benchmarks.AbsabankImmBenchmark'),
            locate('scandeval.datasets.load_absabank_imm')),
        ('sdt-pos', 'the POS part of SDT',
            locate('scandeval.benchmarks.SdtPosBenchmark'),
            locate('scandeval.datasets.load_sdt_pos')),
        ('sdt-dep', 'the DEP part of SDT',
            locate('scandeval.benchmarks.SdtDepBenchmark'),
            locate('scandeval.datasets.load_sdt_dep')),
        ('suc3', 'SUC 3.0',
            locate('scandeval.benchmarks.Suc3Benchmark'),
            locate('scandeval.datasets.load_suc3')),
        ('idt-pos', 'the POS part of IDT',
            locate('scandeval.benchmarks.IdtPosBenchmark'),
            locate('scandeval.datasets.load_idt_pos')),
        ('idt-dep', 'the DEP part of IDT',
            locate('scandeval.benchmarks.IdtDepBenchmark'),
            locate('scandeval.datasets.load_idt_dep')),
        ('mim-gold-ner', 'MIM-GOLD-NER',
            locate('scandeval.benchmarks.MimGoldNerBenchmark'),
            locate('scandeval.datasets.load_mim_gold_ner')),
        ('wikiann-fo', 'the Faroese part of WikiANN',
            locate('scandeval.benchmarks.WikiannFoBenchmark'),
            locate('scandeval.datasets.load_wikiann_fo')),
        ('fdt-pos', 'the POS part of FDT',
            locate('scandeval.benchmarks.FdtPosBenchmark'),
            locate('scandeval.datasets.load_fdt_pos')),
        ('fdt-dep', 'the DEP part of FDT',
            locate('scandeval.benchmarks.FdtDepBenchmark'),
            locate('scandeval.datasets.load_fdt_dep')),
        ('norec-is', 'NoReC-IS',
            locate('scandeval.benchmarks.NorecISBenchmark'),
            locate('scandeval.datasets.load_norec_is')),
        ('norec-fo', 'NoReC-FO',
            locate('scandeval.benchmarks.NorecFOBenchmark'),
            locate('scandeval.datasets.load_norec_fo')),
    ]


PT_CLS = {'token-classification': AutoModelForTokenClassification,
          'text-classification': AutoModelForSequenceClassification}
TF_CLS = {'token-classification': TFAutoModelForTokenClassification,
          'text-classification': TFAutoModelForSequenceClassification}
JAX_CLS = {'token-classification': FlaxAutoModelForTokenClassification,
           'text-classification': FlaxAutoModelForSequenceClassification}
MODEL_CLASSES = dict(pytorch=PT_CLS, tensorflow=TF_CLS, jax=JAX_CLS)


class TwolabelTrainer(Trainer):
    '''Trainer class which deals with two labels.'''
    def __init__(self, split_point: int, **kwargs):
        self.split_point = split_point
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        labels1 = labels[:, :, 0]
        labels2 = labels[:, :, 1]
        labels2 = torch.where(labels2 > 0, labels2 - self.split_point, labels2)

        outputs = model(**inputs)
        logits = outputs.logits

        logits1 = logits[:, :, :self.split_point]
        logits2 = logits[:, :, self.split_point:]
        num_classes2 = logits2.size(2)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss1 = loss_fct(logits1.view(-1, self.split_point),
                         labels1.view(-1))
        loss2 = loss_fct(logits2.view(-1, num_classes2),
                         labels2.view(-1))
        loss = loss1 + loss2
        return (loss, outputs) if return_outputs else loss


class InvalidBenchmark(Exception):
    def __init__(self, message: str = 'This model cannot be benchmarked '
                                      'on the given dataset.'):
        self.message = message
        super().__init__(self.message)


def is_module_installed(module: str) -> bool:
    '''Check if a module is installed.

    Args:
        module (str): The name of the module.

    Returns:
        bool: Whether the module is installed or not.
    '''
    installed_modules_with_versions = list(pkg_resources.working_set)
    installed_modules = [re.sub('[0-9. ]', '', str(module))
                         for module in installed_modules_with_versions]
    installed_modules_processed = [module.lower().replace('-', '_')
                                   for module in installed_modules]
    return module.lower() in installed_modules_processed


def block_terminal_output():
    '''Blocks libraries from writing output to the terminal'''

    # Ignore miscellaneous warnings
    warnings.filterwarnings('ignore',
                            module='torch.nn.parallel*',
                            message=('Was asked to gather along dimension 0, '
                                     'but all input tensors were scalars; '
                                     'will instead unsqueeze and return '
                                     'a vector.'))
    warnings.filterwarnings('ignore', module='seqeval*')

    logging.getLogger('filelock').setLevel(logging.ERROR)

    # Disable the tokenizer progress bars
    ds_logging.get_verbosity = lambda: ds_logging.NOTSET

    # Disable most of the `transformers` logging
    tf_logging.set_verbosity_error()


class DocInherit(object):
    '''Docstring inheriting method descriptor.

    The class itself is also used as a decorator.
    '''
    def __init__(self, mthd: Callable):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError(f'Can\'t find "{self.name}" in parents')
        func.__doc__ = source.__doc__
        return func


doc_inherit = DocInherit
