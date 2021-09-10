<div align='center'>

<img src="https://raw.githubusercontent.com/saattrupdan/ScandEval/main/gfx/scandeval.png" width="517" height="217">

### Evaluation of language models on mono- or multilingual Scandinavian language tasks.

______________________________________________________________________
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/ScandEval)](https://github.com/saattrupdan/ScandEval/commits/main)
[![ReadTheDocs](https://readthedocs.org/projects/scandeval/badge/?version=latest)](https://scandeval.readthedocs.io/en/latest/?badge=latest)
[![PyPI Status](https://badge.fury.io/py/scandeval.svg)](https://badge.fury.io/py/scandeval)
[![License](https://img.shields.io/github/license/saattrupdan/ScandEval)](https://github.com/saattrupdan/ScandEval/blob/main/LICENSE)


</div>

## Installation
To install the package simply write the following command in your favorite
terminal:
```shell
$ pip install scandeval[all]
```

This will install all the model frameworks currently supported (`pytorch`,
`tensorflow`, `jax` and `spacy`). If you know you only need one of these, you
can install a slimmer package like so:
```shell
$ pip install scandeval[pytorch]
```

Lastly, if you are not interesting in benchmarking models, but just want to
use the package to download datasets, then the following command will do the
trick:
```shell
$ pip install scandeval
```

## Quickstart
### Benchmarking from the Command Line
The easiest way to benchmark models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:
```shell
$ scandeval --model_id <model_id>
```

Here `model_id` is the HuggingFace model ID, which can be found on the
[HuggingFace Hub](https://huggingface.co/models). By default this will
benchmark the model on all the datasets eligible. If you want to benchmark on a
specific dataset, this can be done via the `--dataset` flag. This will for
instance evaluate the model on the `AngryTweets` dataset:
```shell
$ scandeval --model_id <model_id> --dataset angry-tweets
```

We can also separate by language. To benchmark all Danish models, say, this can
be done using the `language` tag, like so:
```shell
$ scandeval --language da
```

Multiple models, datasets and/or languages can be specified by just attaching
multiple arguments. Here is an example with two models:
```shell
$ scandeval --model_id <model_id1> --model_id <model_id2> --dataset angry-tweets
```

See all the arguments and options available for the `scandeval` command by
typing
```shell
$ scandeval --help
```

### Benchmarking from a Script
In a script, the syntax is similar to the command line interface. You simply
initialise an object of the `Benchmark` class, and call this benchmark object
with your favorite models and/or datasets:
```python
>>> from scandeval import Benchmark
>>> benchmark = Benchmark()
>>> benchmark('<model_id>')
```

To benchmark on a specific dataset, you simply specify the second argument,
shown here with the `AngryTweets` dataset again:
```python
>>> benchmark('<model_id>', 'angry-tweets')
```

This would benchmark all Danish models:
```python
>>> benchmark(language='da')
```

See the [documentation](https://scandeval.readthedocs.io/en/latest/) for a more
in-depth description.


### Downloading Datasets
If you are just interested in downloading a dataset rather than benchmarking,
this can be done as follows:
```python
>>> from scandeval import load_dataset
>>> X_train, X_test, y_train, y_test = load_dataset('angry-tweets')
```

Here `X_train` and `X_test` will be Pandas dataframes containing the relevant
texts, and `y_train` and `y_test` will be Pandas dataframes containing the
associated labels.

See the [documentation](https://scandeval.readthedocs.io/en/latest/) for a list
of all the datasets that can be loaded.


## Documentation
The full documentation can be found on
[ReadTheDocs](https://scandeval.readthedocs.io/en/latest).


## Citing ScandEval
If you want to cite the framework then feel free to use this:
```bibtex
@article{nielsen2021scandeval,
  title={ScandEval: Evaluation of language models on mono- or multilingual Scandinavian language tasks.},
  author={Nielsen, Dan Saattrup},
  journal={GitHub. Note: https://github.com/saattrupdan/ScandEval},
  year={2021}
}
```

## Remarks
The image used in the logo has been created by the amazing [Scandinavia and the
World](https://satwcomic.com/) team. Go check them out!
