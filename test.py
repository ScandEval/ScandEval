'''Testing script'''

import logging
import warnings

def test_dane():
    from scandeval import DaneEvaluator
    dane_eval = DaneEvaluator(cache_dir='/media/secure/dan/huggingface')
    dane_eval.evaluate('Maltehb/-l-ctra-danish-electra-small-cased',
                       num_finetunings=10)

if __name__ == '__main__':
    test_dane()
