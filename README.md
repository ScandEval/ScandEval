<div align='center'>
<img src="https://raw.githubusercontent.com/saattrupdan/ScandEval/main/gfx/scandeval.png" width="517" height="217">
</div>

### Evaluation of pretrained language models on mono- or multilingual Scandinavian language tasks.

______________________________________________________________________
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/ScandEval)](https://github.com/saattrupdan/ScandEval/commits/main)
[![ReadTheDocs](https://readthedocs.org/projects/scandeval/badge/?version=latest)](https://scandeval.readthedocs.io/en/latest/?badge=latest)
[![PyPI Status](https://badge.fury.io/py/scandeval.svg)](https://badge.fury.io/py/scandeval)
[![License](https://img.shields.io/github/license/saattrupdan/ScandEval)](https://github.com/saattrupdan/ScandEval/blob/main/LICENSE)


## Installation
To install the package simply write the following command in your favorite terminal:
```
$ pip install scandeval
```

## Quickstart
### Benchmarking from the Command Line
The easiest way to benchmark pretrained models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:
```
$ scandeval --model_id <model_id>
```

Here `model_id` is the HuggingFace model ID, which can be found on the [HuggingFace
Hub](https://huggingface.co/models). By default this will benchmark the model on all
the datasets eligible. If you want to benchmark on a specific dataset, this can be done
via the `--dataset` flag. This will for instance evaluate the model on the
`AngryTweets` dataset:
```
$ scandeval --model_id <model_id> --dataset angry-tweets
```

We can also separate by language. To benchmark all Danish models on all Danish
datasets, say, this can be done using the `language` tag, like so:
```
$ scandeval --language da
```

Multiple models, datasets and/or languages can be specified by just attaching multiple
arguments. Here is an example with two models:
```
$ scandeval --model_id <model_id1> --model_id <model_id2> --dataset angry-tweets
```

The specific model version to use can also be added after the suffix '@':
```
$ scandeval --model_id <model_id>@<commit>
```

It can be a branch name, a tag name, or a commit id. It defaults to 'main' for latest.

See all the arguments and options available for the `scandeval` command by typing
```
$ scandeval --help
```

### Benchmarking from a Script
In a script, the syntax is similar to the command line interface. You simply initialise
an object of the `Benchmarker` class, and call this benchmark object with your favorite
models and/or datasets:
```
>>> from scandeval import Benchmarker
>>> benchmark = Benchmarker()
>>> benchmark('<model_id>')
```

To benchmark on a specific dataset, you simply specify the second argument, shown here
with the `AngryTweets` dataset again:
```
>>> benchmark('<model_id>', 'angry-tweets')
```

This would benchmark all Nynorsk models on Nynorsk datasets:
```
>>> benchmark(language='nn')
```

See the [documentation](https://scandeval.readthedocs.io/en/latest/) for a more
in-depth description.


## Citing ScandEval
If you want to cite the framework then feel free to use this:
```
@article{nielsen2022scandeval,
  title={ScandEval: Evaluation of language models on mono- or multilingual Scandinavian language tasks.},
  author={Nielsen, Dan Saattrup},
  journal={GitHub. Note: https://github.com/saattrupdan/ScandEval},
  year={2022}
}
```

## Remarks
The image used in the logo has been created by the amazing [Scandinavia and the
World](https://satwcomic.com/) team. Go check them out!


## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── README.md
├── docs
├── makefile
├── notebooks
│   └── scandeval-truncation.ipynb
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scandeval
│   │   ├── __init__.py
│   │   ├── benchmark_dataset.py
│   │   ├── benchmarker.py
│   │   ├── callbacks.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── configs
│   │   │   ├── datasets
│   │   │   │   ├── absabank-imm.yaml
│   │   │   │   ├── angry-tweets.yaml
│   │   │   │   ├── dane.yaml
│   │   │   │   ├── mim-gold-ner.yaml
│   │   │   │   ├── norec.yaml
│   │   │   │   ├── norne-nb.yaml
│   │   │   │   ├── norne-nn.yaml
│   │   │   │   ├── scala-da.yaml
│   │   │   │   ├── scala-fo.yaml
│   │   │   │   ├── scala-is.yaml
│   │   │   │   ├── scala-nb.yaml
│   │   │   │   ├── scala-nn.yaml
│   │   │   │   ├── scala-sv.yaml
│   │   │   │   ├── suc3.yaml
│   │   │   │   └── wikiann-fo.yaml
│   │   │   ├── language_codes.yaml
│   │   │   └── tasks
│   │   │       ├── la.yaml
│   │   │       ├── ner.yaml
│   │   │       ├── qa.yaml
│   │   │       └── sent.yaml
│   │   ├── dataset_factory.py
│   │   ├── datasets.py
│   │   ├── exceptions.py
│   │   ├── hf_hub.py
│   │   ├── ner.py
│   │   ├── qa.py
│   │   ├── scores.py
│   │   ├── text_classification.py
│   │   ├── training_args_with_mps_support.py
│   │   └── utils.py
│   └── scripts
│       ├── create_absabank_imm.py
│       ├── create_angry_tweets.py
│       ├── create_dane.py
│       ├── create_mim_gold_ner.py
│       ├── create_norec.py
│       ├── create_norne.py
│       ├── create_scala.py
│       ├── create_scandiqa.py
│       ├── create_suc3.py
│       ├── create_wikiann_fo.py
│       ├── fix_dot_env_file.py
│       └── load_ud_pos.py
└── tests
    └── __init__.py
```
