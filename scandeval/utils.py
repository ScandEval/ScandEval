'''Utility functions to be used in other scripts'''

from functools import wraps
from typing import Callable
import warnings
import datasets.utils.logging as ds_logging
import transformers.utils.logging as tf_logging
from transformers import (AutoModelForTokenClassification,
                          AutoModelForSequenceClassification,
                          AutoModelForQuestionAnswering,
                          TFAutoModelForTokenClassification,
                          TFAutoModelForSequenceClassification,
                          TFAutoModelForQuestionAnswering,
                          FlaxAutoModelForTokenClassification,
                          FlaxAutoModelForSequenceClassification,
                          FlaxAutoModelForQuestionAnswering)


MODEL_CLASSES = {
    'pytorch': {'token-classification': AutoModelForTokenClassification,
                'text-classification': AutoModelForSequenceClassification,
                'question-answering': AutoModelForQuestionAnswering},
    'tensorflow': {'token-classification': TFAutoModelForTokenClassification,
                   'text-classification': TFAutoModelForSequenceClassification,
                   'question-answering': TFAutoModelForQuestionAnswering},
    'jax': {'token-classification': FlaxAutoModelForTokenClassification,
            'text-classification': FlaxAutoModelForSequenceClassification,
            'question-answering': FlaxAutoModelForQuestionAnswering}
}


def block_terminal_output():
    '''Blocks libraries from writing output to the terminal'''

    # Ignore miscellaneous warnings
    warnings.filterwarnings(
        'ignore',
        module='torch.nn.parallel*',
        message=('Was asked to gather along dimension 0, but all input '
                 'tensors were scalars; will instead unsqueeze and return '
                 'a vector.')
    )
    warnings.filterwarnings('ignore', module='seqeval*')

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

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden: break

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
