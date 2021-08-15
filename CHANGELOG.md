# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Always ensure that a model can deal with the labels in the dataset when
  finetuning. If the model has not been trained on the label, then this will
  result in the model always getting that label wrong. For instance, this is
  the case for finetuned NER models not having been trained on MISC tags, if
  they are being evaluated on the DaNE dataset.

### Fixed
- Fixed bug when evaluating SpaCy models.
- Only removing objects at memory cleanup if they exist at all.


## [v0.6.0] - 2021-08-15
### Added
- When finetuning models, 10% of the training data is used to evaluate the
  models, which is used to choose the best performing model across all the
  epochs trained. This will allow for a more fair comparison, as some models
  degrade over time, while other models need a longer time to train.

### Changed
- Uniformised the `_log_metrics` method for all benchmarks, now only defined in
  `BaseBenchmark`.

### Fixed
- Garbage collects when downsizing batch size, to not keep all the previous
  models in memory.
- Typos in logging.


## [v0.5.2] - 2021-08-13
### Fixed
- Fixed bug when `evaluate_train` was set to False.


## [v0.5.1] - 2021-08-13
### Fixed
- The bootstrapping of the datasets is now done properly. Previously the
  bootstrapped datasets were not converted to HuggingFace Dataset objects.


## [v0.5.0] - 2021-08-12
### Added
- It is possible to only evaluate on the test sets, to save some time. This can
  be done in the `Benchmark` class using the `evaluate_train` argument, and in
  the CLI with the `--evaluate_train` flag.
- Added `progress_bar` argument to `Benchmark` to control whether progress bars
  should be shown, and added the `no_progress_bar` flag to the CLI for the same
  reason.

### Changed
- Updated `epochs` and `warmup_steps` of all the datasets to something more
  reasonable, enabling better comparisons of the finetuned models.
- Changed calculation of confidence intervals, which is now based on
  bootstrapping rather than the analytic approach. It will now evaluate ten
  times on the test set and compute a bootstrap estimate of the standard error,
  which is uses to compute an interval around the score on the entire test set.


## [v0.4.3] - 2021-08-12
### Fixed
- RuntimeErrors occuring during training will now raise an `InvalidBenchmark`
  exception, which means that the CLI and the `Benchmark` class will skip it.
  This is for instance caused when `max_length` has not been specified in the
  model config, meaning that the tokeniser does not know how much to truncate.


## [v0.4.2] - 2021-08-12
### Fixed
- Now catching the error where tokenisation is not possible, due to the model
  having been trained on a different task than what is present in the dataset.
  E.g., if a generator model is trained on a classification task.


## [v0.4.1] - 2021-08-12
### Fixed
- Now catching the error when the model's config does not align with the model
  class. When using the CLI or `Benchmark`, these will be skipped.


## [v0.4.0] - 2021-08-11
### Added
- Added confidence intervals for finetuned models, where there is a 95%
  likelihood that the true score would belong to the interval, given infinite
  data from the same distribution. In the case of "raw" pretrained models, this
  radius is added onto the existing interval, so that both the uncertainty in
  model initialisation as well as sample size of the validation dataset affects
  the size of the interval.
- Added garbage collection after each benchmark, which will (hopefully) prevent
  memory leaking when benchmarking several models.

### Changed
- New logo, including the Faroe Islands!
- Allow the possibility to include all languages and/or tasks in the CLI and
  the `Benchmark` class.
- Added Icelandic and Faroese to default list of languages in CLI and the
  `Benchmark` class.
- The default value for `task` is now all tasks, which also includes models
  that haven't been assigned any task on the HuggingFace Hub;
- If a model cannot be trained without running out of CUDA memory, even with a
  batch size of 1, then the model will be skipped in `Benchmark` and the CLI.

### Fixed
- New model is initialised if CUDA runs out of memory, to ensure that we are
  now continuing to train the previous model.
- Dependency parsing now implemented properly as two-label classification, with
  associated UAS and LAS metric computations. Works for pretrained SpaCy models
  as well as finetuning general language models.


## [v0.3.1] - 2021-08-10
### Fixed
- Reduces batch size if CUDA runs out of memory during evaluation.
- Loading of text classification datasets now working properly.


## [v0.3.0] - 2021-08-10
### Changed
- The `W036` warning message from SpaCy is no longer shown.

### Fixed
- Raise `InvalidBenchmark` if model cannot be loaded from the HuggingFace Hub.


## [v0.2.0] - 2021-08-09
### Added
- Added the part-of-speech tagging task from the Danish Dependency Treebank.
  Can be loaded with `load_ddt_pos` and used in `Benchmark` as `ddt-pos`.
- Added the dependency parsing task from the Danish Dependency Treebank.
  Can be loaded with `load_ddt_ddt` and used in `Benchmark` as `ddt-dep`.
- Documentation section and link to `README`
- The `Benchmark` class and the CLI now accepts a `batch_size` argument

### Changed
- `Benchmark` arguments `languages`, `tasks`, `model_ids` and `datasets` have
  been renamed to `language`, `task`, `model_id` and `dataset`, to keep it
  consistent with the CLI.
- When loading datasets, these will now be four dictionaries instead of lists,
  to allow for distinguishing features and labels.
- `batch_size` arguments can now only be among 1, 2, 4, 8, 16 and 32, and the
  corresponding gradient accumulation will be set to 32, 16, 8, 4, 2 and 1,
  respectively. This is to ensure that all finetuning is done using the same
  effective batch size, to ensure fair comparisons.
- Batch sizes are automatically halved if the GPU runs out of memory, with
  gradient accumulation correspondingly doubles.
- Evaluation of `SpaCy` models on token classification tasks are more accurate.

### Fixed
- `README` typos fixed, and image renders correctly


## [v0.1.0] - 2021-08-05
### Added
- First beta release
- Features Danish sentiment, hate speech detection and named entitity
  recognition datasets for benchmarking
