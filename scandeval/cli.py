'''Command-line interface for benchmarking'''

import click
from typing import Tuple

from .benchmark import Benchmark


@click.command()
@click.option('model_id', '-m',
              default=None,
              show_default=True,
              multiple=True,
              help='The HuggingFace model ID of the model(s) to be '
                   'benchmarked. If not specified then all models will be '
                   'benchmarked, filtered by `language` and `task`.')
@click.option('--dataset', '-d',
              default=None,
              show_default=True,
              multiple=True,
              type=click.Choice(['dane', 'dane-no-misc', 'ddt-pos', 'ddt-dep',
                                 'angry-tweets', 'twitter-sent', 'dkhate',
                                 'europarl1', 'europarl2', 'lcc1', 'lcc2']),
              help='The name of the benchmark dataset. If not specified then '
                   'all datasets will be benchmarked.')
@click.option('--language', '-l',
              default=['da', 'sv', 'no'],
              show_default=True,
              multiple=True,
              type=click.Choice(['da', 'sv', 'no']),
              help='The languages to benchmark. Only relevant if `model_id` '
                   'is not specified.')
@click.option('--task', '-t',
              default=('fill-mask',
                       'token-classification',
                       'text-classification'),
              show_default=True,
              multiple=True,
              type=click.Choice(['fill-mask',
                                 'token-classification',
                                 'text-classification']),
              help='The tasks to benchmark. Only relevant if `model_id` '
                   'is not specified.')
@click.option('--num_finetunings', '-n',
              default=10,
              show_default=True,
              help='The number of times a language model should be '
                   'finetuned.')
@click.option('--batch_size', '-b',
              default=32,
              type=int,
              show_default=True,
              help=('The batch size used to finetune the models. This '
                    'have to be among 1, 2, 4, 8, 16 and 32.'))
@click.option('--verbose', '-v',
              is_flag=True,
              show_default=True,
              help='Whether extra input should be outputted during '
                   'benchmarking')
def benchmark(model_id: Tuple[str],
              dataset: Tuple[str],
              language: Tuple[str],
              task: Tuple[str],
              num_finetunings: int,
              batch_size: int,
              verbose: bool = False):
    '''Benchmark language models on Scandinavian language tasks.'''
    # Initialise the benchmarker class
    benchmarker = Benchmark(language=list(language),
                            task=list(task),
                            batch_size=batch_size,
                            verbose=verbose)

    # Perform the benchmark evaluation
    model_id = None if len(model_id) == 0 else list(model_id)
    dataset = None if len(dataset) == 0 else list(dataset)
    benchmarker(model_id=model_id,
                dataset=dataset,
                num_finetunings=num_finetunings,
                save_results=True)
