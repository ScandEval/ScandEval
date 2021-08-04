<div align='center'>

<img src="gfx/scandeval.png" width="370" height="200">

### Evaluation of language models on mono- or multilingual Scandinavian language tasks.

______________________________________________________________________

</div>

## Installation
To install the package simply write the following command in your favorite
terminal:
```shell
$ pip install scandeval[all]
```

This will install all the model frameworks currently supported (`pytorch`,
`tensorflow`, `flax` and `spacy`). If you know you only need one of these, you
can install the slimmer package like so:
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
### In the Command Line
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

If you want to benchmark _all_ Danish models, this can be done using the
`languages` tag, like so:
```shell
$ scandeval --languages da
```

See all the arguments and options available for the `scandeval` command by
typing
```shell
$ scandeval --help
```

### In a Script
In a script, the syntax is similar to the command line interface. You simply
initialise an object of the `Benchmark` class, and call this benchmark object
with your favorite models and/or datasets:
```python
>>> from scandeval import Benchmark
>>> benchmark = Benchmark()
>>> benchmark('<model_id>')
```

To benchmark on a specific dataset, you simply specify the second argument:
```python
>>> benchmark('<model_id>', 'angry-tweets')
```

To benchmark _all_ Danish models, this is given at initialisation:
```python
>>> benchmark = Benchmark(languages='da')
>>> benchmark()
```

### Downloading Datasets
If you are just interested in downloading a dataset rather than benchmarking,
this can be done as follows:
```python
>>> from scandeval.datasets import load_angry_tweets
>>> X_train, X_test, y_train, y_test = load_angry_tweets()
```

Here `X_train` and `X_test` will be lists containing the relevant texts, and
`y_train` and `y_test` will be lists containing the associated labels.


## Citing ScandEval
If you want to cite the framework then feel free to use this:
```bibtex
@article{nielsen2021scandeval,
  title={ScandEval},
  author={Nielsen, Dan Saattrup},
  journal={GitHub. Note: https://github.com/saattrupdan/ScandEval},
  year={2021}
}
```

### Remarks
The image used in the logo has been created by the amazing [Scandinavia and the World](https://satwcomic.com/) team. Go check them out!
