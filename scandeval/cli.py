'''Command-line interface for benchmarking'''

import click
from typing import Tuple

from .benchmark import Benchmark
from .utils import get_all_datasets


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
              type=click.Choice([n for n, _, _, _ in get_all_datasets()]),
              help='The name of the benchmark dataset. If not specified then '
                   'all datasets will be benchmarked.')
@click.option('--language', '-l',
              default=['da', 'sv', 'no', 'nb', 'nn', 'is', 'fo'],
              show_default=True,
              multiple=True,
              type=click.Choice(['da', 'sv', 'no', 'nb', 'nn', 'is', 'fo']),
              help='The languages to benchmark. Only relevant if `model_id` '
                   'is not specified.')
@click.option('--task', '-t',
              default=['all'],
              show_default=True,
              multiple=True,
              type=click.Choice(['all',
                                 'fill-mask',
                                 'token-classification',
                                 'text-classification']),
              help='The tasks to benchmark. Only relevant if `model_id` '
                   'is not specified.')
@click.option('--evaluate_train',
              is_flag=True,
              show_default=True,
              help='Whether the training set should be evaluated.')
@click.option('--no_progress_bar', '-p',
              is_flag=True,
              show_default=True,
              help='Whether progress bars should be shown.')
@click.option('--verbose', '-v',
              is_flag=True,
              show_default=True,
              help='Whether extra input should be outputted during '
                   'benchmarking')
def benchmark(model_id: Tuple[str],
              dataset: Tuple[str],
              language: Tuple[str],
              task: Tuple[str],
              evaluate_train: bool,
              no_progress_bar: bool,
              verbose: bool = False):
    '''Benchmark language models on Scandinavian language tasks.'''
    # Initialise the benchmarker class
    benchmarker = Benchmark(language=list(language),
                            task='all' if 'all' in task else list(task),
                            progress_bar=(not no_progress_bar),
                            save_results=True,
                            evaluate_train=evaluate_train,
                            verbose=verbose)

    # Perform the benchmark evaluation
    model_id = None if len(model_id) == 0 else list(model_id)
    dataset = None if len(dataset) == 0 else list(dataset)
    benchmarker(model_id=model_id, dataset=dataset)
