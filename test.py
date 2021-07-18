'''Testing script'''

from scandeval import DaneEvaluator


def test_dane():
    dane_eval = DaneEvaluator(cache_dir='/media/secure/dan/huggingface')
    dane_eval.evaluate('Maltehb/-l-ctra-danish-electra-small-cased')


if __name__ == '__main__':
    test_dane()
