'''Command-line interface for benchmarking'''

import click
from typing import Tuple

from .benchmark import Benchmark
from .utils import get_all_datasets


@click.command()
@click.option('--model-id', '-m',
              default=None,
              show_default=True,
              multiple=True,
              help='The HuggingFace model ID of the model(s) to be '
                   'benchmarked. If not specified then all models will be '
                   'benchmarked, filtered by `language` and `task`. '
                   'The specific model version to use can be added after '
                   'the suffix "@": "<model_id>@v1.0.0". It can be a branch '
                   'name, a tag name, or a commit id (currently only '
                   'supported for HuggingFace models, and it defaults to '
                   '"main" for latest).')
@click.option('--dataset', '-d',
              default=None,
              show_default=True,
              multiple=True,
              type=click.Choice([n for n, _, _, _ in get_all_datasets()]),
              help='The name of the benchmark dataset. If not specified then '
                   'all datasets will be benchmarked.')
@click.option('--language', '-l',
              default=['da', 'sv', 'no'],
              show_default=True,
              multiple=True,
              type=click.Choice(['da', 'sv', 'no', 'nb', 'nn', 'is', 'fo']),
              help='The languages to benchmark, both for models and datasets. Only '
                   'relevant if `model-id` and `dataset` have not both been specified.')
@click.option('--model-language', '-ml',
              default=None,
              show_default=True,
              multiple=True,
              type=click.Choice(['da', 'sv', 'no', 'nb', 'nn', 'is', 'fo']),
              help='The model languages to benchmark. Only relevant if `model-id` has '
                   'not been specified. If specified then this will override '
                   '`language` for models.')
@click.option('--dataset-language', '-dl',
              default=None,
              show_default=True,
              multiple=True,
              type=click.Choice(['da', 'sv', 'no', 'nb', 'nn', 'is', 'fo']),
              help='The dataset languages to benchmark. Only relevant if `dataset` has '
                   'not been specified. If specified then this will override '
                   '`language` for datasets.')
@click.option('--task', '-t',
              default=['all'],
              show_default=True,
              multiple=True,
              type=click.Choice(['all',
                                 'fill-mask',
                                 'token-classification',
                                 'text-classification']),
              help='The tasks to benchmark. Only relevant if `model-id` '
                   'is not specified.')
@click.option('--evaluate-train',
              is_flag=True,
              show_default=True,
              help='Whether the training set should be evaluated.')
@click.option('--no-progress-bar', '-p',
              is_flag=True,
              show_default=True,
              help='Whether progress bars should be shown.')
@click.option('--raise-error-on-invalid-model', '-r',
              is_flag=True,
              show_default=True,
              help='Whether to raise an error if a model is invalid.')
@click.option('--train-size', '-s',
              default=[1024],
              show_default=True,
              multiple=True,
              type=int,
              help='The number of training samples to train on')
@click.option('--verbose', '-v',
              is_flag=True,
              show_default=True,
              help='Whether extra input should be outputted during '
                   'benchmarking')
def benchmark(model_id: Tuple[str],
              dataset: Tuple[str],
              language: Tuple[str],
              model_language: Tuple[str],
              dataset_language: Tuple[str],
              raise_error_on_invalid_model: bool,
              train_size: Tuple[int],
              task: Tuple[str],
              evaluate_train: bool,
              no_progress_bar: bool,
              verbose: bool = False):
    '''Benchmark language models on Scandinavian language tasks.'''

    # Initialise the benchmarker class
    benchmarker = Benchmark(language=list(language),
                            model_language=list(model_language),
                            dataset_language=list(dataset_language),
                            task='all' if 'all' in task else list(task),
                            progress_bar=(not no_progress_bar),
                            save_results=True,
                            evaluate_train=evaluate_train,
                            raise_error_on_invalid_model=raise_error_on_invalid_model,
                            verbose=verbose,
                            train_size=list(train_size))

    # Perform the benchmark evaluation
    model_id = None if len(model_id) == 0 else list(model_id)
    dataset = None if len(dataset) == 0 else list(dataset)
    benchmarker(model_id=model_id, dataset=dataset)
