# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Changed
- Now also outputting aggregated metrics in the resulting
  `scandeval_benchmark_results.json` file. This `json` file now has keys
  `raw_metrics` and `total`, with `raw_metrics` containing the previous (raw)
  scores, and the value of the new `total` key has aggregated scores (means and
  standard errors).


## [v1.3.8] - 2021-11-25
### Changed
- All training/evaluation progress bars are now removed when they are finished,
  and the training progress bar has no total anymore, as it was misleading.


## [v1.3.7] - 2021-11-25
### Fixed
- Removed `transformers` logging during evaluation as well.


## [v1.3.6] - 2021-11-25
### Changed
- Now only updating the list of benchmarks in the `Benchmark` during
  initialisation, and also logs it. This should make subsequent calls to the
  `benchmark` method faster.

### Fixed
- Removed `transformers` logging properly.


## [v1.3.5] - 2021-11-23
### Fixed
- Set the number of warmup steps to be the intended one training set pass,
  where previously it was effectively 8x that amount, due to gradient
  accumulation.
- Added the NER label synonyms `OBJORG=ORG`, `LOCPRS=LOC`, `LOCORG=LOC` and
  `ORGPRS=ORG`.
- Explicitly added `numpy` to the `install_requires` list. This is normally not
  a problem, as it's a requirement for other required packages, but this
  depends on the order in which the requirements are installed. This avoids
  such errors caused by misordering the requirements.


## [v1.3.4] - 2021-11-11
### Fixed
- Indexing error during synonym setup of finetuned models.


## [v1.3.3] - 2021-11-11
### Fixed
- When a finetuned model has labels which are synonyms of each other, they are
  now properly treated as synonyms, where previously this caused the model to
  have misaligned `id2label` and `label2id` conversion dictionaries.


## [v1.3.2] - 2021-11-11
### Fixed
- Added the NER label synonyms `GPE_LOC=LOC`, `GPE_ORG=ORG`, `LOC/ORG=LOC`,
  `ORG/PRS=ORG`, `OBJ/ORG=ORG`, as Norwegian and Swedish models tend to use
  these.


## [v1.3.1] - 2021-11-11
### Fixed
- Fixed a bug in label synonyms when benchmarking a finetuned spaCy for NER.


## [v1.3.0] - 2021-11-11
### Added
- Added label synonyms for NER benchmarking, which will enforce a more fair
  comparison of finetuned NER models, if the models have been trained on
  datasets with different labelling (e.g., `Person` instead of `PER`).


## [v1.2.1] - 2021-11-11
### Removed
- Properly removed the Icelandic WikiANN-IS data files. It was removed from the
  package, but the underlying files were still lying in the repository.


## [v1.2.0] - 2021-10-15
### Added
- Added the Icelandic NER dataset MIM-GOLD-NER. This can now be loaded as
  `mim-gold-ner` in the `Benchmark` class and through the CLI.

### Removed
- Removed the Icelandic WikiANN-IS dataset, as this has now been replaced by
  the MIM-GOLD-NER dataset.


## [v1.1.3] - 2021-10-04
### Fixed
- Added truncation and padding when tokenising token classification datasets.


## [v1.1.2] - 2021-09-27
### Fixed
- Missing dependency parsing tags.


## [v1.1.1] - 2021-09-27
### Fixed
- Reduce validation batch size if CUDA runs out of memory, rather than only
  reducing training batch size.


## [v1.1.0] - 2021-09-13
### Added
- Added Icelandic and Faroese translations of the Norwegian `NoReC` sentiment
  analysis dataset. These can be loaded as `norec-is` and `norec-fo`,
  respectively.

### Changed
- When loading datasets with `load_dataset`, the result is now four dataframes,
  rather than dictionaries. As the data can be accessed in the same way as with
  dictionaries, this maintains backwards compatibility.
- If a finetuned NER model has been trained on NER tags not present amongst the
  ones in the dataset, then these are either converted to `MISC` tags (if these
  are present in the dataset) and otherwise `O` tags. This will make the
  benchmarking of finetuned diverse NER models more fair.

### Fixed
- There was an error when a SpaCy model was benchmarked on a dataset that it
  was not trained on. It now raises an appropriate `InvalidBenchmark`
  exception, and will be skipped in the CLI and with the `Benchmark` class.


## [v1.0.2] - 2021-09-09
### Fixed
- Replaced abbreviations with spaces, such as "o s v" in the SDT corpus, with
  their proper version "o.s.v.".


## [v1.0.1] - 2021-09-09
### Fixed
- The URLs for the `wikiann-is` and `wikiann-fo` were wrong and have been
  corrected.


## [v1.0.0] - 2021-09-09
### Added
- Added the Icelandic and Faroese WikiANN datasets, for NER evaluation. They
  can be loaded as `wikiann-is` and `wikiann-fo` in the CLI and via the
  `Benchmark` class.
- Added the Icelandic and Faroese parts of the Universal Dependencies datasets,
  containing POS and dependency parsing tags. They can be loaded as `idt-pos`,
  `idt-dep`, `fdt-pos` and `fdt-dep`, respectively.


## [v0.17.0] - 2021-09-09
### Added
- Added the Dataset for Linguistic Acceptability Judgments (DaLaJ) dataset,
  which is here used as a binary classification dataset, in which sentences
  have to be classified as correct Swedish or not. It can be loaded as `dalaj`
  in the CLI and via the `Benchmark` class.
- Added the ABSAbank-Imm dataset, which is an aspect-based sentiment analysis
  dataset in Swedish, namely, the sentiment towards immigration. The original
  dataset featured a floating point score between 0 and 5, which has been
  reduced to a classifical three-way classification (`negative`, `neutral` and
  `positive`). It can be loaded as `absabank-imm` in the CLI and via the
  `Benchmark` class.
- Added the POS and dependency parsing parts of the Swedish Dependency Treebank
  (SDT). They can be loaded as `sdt-pos` and `sdt-dep` in the CLI and via the
  `Benchmark` class.
- Added the Stockholm-Umeå corpus 3.0 (SUC 3.0), a Swedish NER dataset. It can
  be loaded as `suc3` in the CLI and via the `Benchmark` class.
- Added abstract `NerBenchmark`, `PosBenchmark` and `DepBenchmark` classes, to
  ensure uniformity.

### Changed
- Uniformised all the NER datasets. They now all only have the NER tags `PER`,
  `LOC`, `ORG` and `MISC`.
- Uniformised all the dependency parsing datasets. They now all only have the
  main dependency parsing tags, without the subtags (so `acl:cleft` has been
  changed to `acl`, for instance).
- Changed the columns in all text classification datasets to `text` and
  `label`, to make it more uniform.


## [v0.16.0] - 2021-09-07
### Fixed
- Upped the number index tokens for dependency parsing from 100 to 512. This
  will need to be done better in the future, but is a fix for now.

### Added
- Added the random models `random-roberta-sequence-clf` and
  `random-roberta-token-clf` to the default list of model IDs when benchmarking
  all models.


## [v0.15.1] - 2021-09-03
### Fixed
- The list of dependency tags in the `ndt-nb-dep` and `ndt-nn-dep` were wrong.
  They have now been changed to all the tags occurring in the training sets.
- The `europarl_sent` data folder has now been renamed to `europarl`, so that
  it can be loaded correctly with `load_dataset`.


## [v0.15.0] - 2021-09-02
### Added
- Added the Bokmål and Nynorsk POS and DEP parts of the Norwegian Dependency
  Treebank dataset (NDT). They can be loaded as `ndt-nb-pos`, `ndt-nn-pos`,
  `ndt-nb-dep` and `ndt-nn-dep`, respectively, from the CLI and the `Benchmark`
  class.

### Removed
- Removed the `EuroparlSubj` and `TwitterSubj` datasets, as they were too easy
  and did not really differentiate models.
- Removed the abstract `SentimentClassificationBenchmark` and
  `BinaryClassificationBenchmark`, to simplify the classes. There is now only
  one `TextClassificationBenchmark`, which always evaluates with macro-F1.

### Changed
- Changed the name of `europarl-sent` to `europarl`, as `europarl-subj` now
  does not exist anymore.
- Changed the `nordial` dataset to the original 4-way classification dataset.


## [v0.14.1] - 2021-09-02
### Fixed
- Remove duplicate model IDs when calling the CLI or `Benchmark` class without
  any specified model IDs.


## [v0.14.0] - 2021-08-31
### Added
- Added the Bokmål and Nynorsk parts of the NorNE dataset, for named entity
  recognition. They can be loaded with the `norne-nb` and `norne-nn` names.
- There is now a `load_dataset` function, which can load any dataset, using the
  dataset's name (same name as in the CLI). For instance,
  `load_dataset('angry-tweets')` loads the `AngryTweets` dataset. This can be
  imported directly from the package: `from scandeval import load_dataset`. The
  individual dataset loading functions can still be imported as before; e.g.,
  `from scandeval.datasets import load_angry_tweets`.

### Changed
- Refactored folder structure with benchmarks and datasets.
- Separated `dane` and `dane-no-misc` into two distinct benchmark classes. The
  `dane-no-misc` can now also be loaded with the `load_dataset` function.


## [v0.13.0] - 2021-08-30
### Added
- Added the Norwegian Review Corpus (NoReC), a sentiment classification dataset
  in Norwegian.
- Added the Bokmål/Nynorsk part of the Norwegian Dialect dataset (NorDial), a
  binary classification dataset in Norwegian.

### Changed
- Changed the early stopping patience to `2 + 1000 // len(train)` from `2 + 250
  // len(train)`, to allow more patience (and thus, more stability), for
  smaller datasets.


## [v0.12.0] - 2021-08-26
### Changed
- Merged the `lcc1` and `lcc2` datasets into one `lcc` dataset, which is
  reasonable as they have been annotated by the same person. The `lcc2` dataset
  was too small to give reasonable benchmarking results.
- Renamed the `europarl2` dataset to `europarl_sent`

### Removed
- Removed the `europarl1` dataset, as it was too small to give reliable
  benchmarking results. This dataset could not simply be added to the
  `europarl2` dataset, as with the new `lcc` dataset, as the annotaters are not
  the same.

### Fixed
- If errors occur during benchmarking, then garbage collect before skipping to
  the next benchmark, to avoid memory issues.


## [v0.11.2] - 2021-08-25
### Fixed
- Issue with `model_max_length` in tokenizer meant that models with an ill-set
  value of `max_position_embeddings` could not be benchmarked. Now, if
  `model_max_length` is not set then the minimal value of the sizes in
  `max_model_input_sizes` will be used (which is usually 512).

### Changed
- Disabling CUDNN benchmark when using the `pytorch` framework, to enforce
  better reproducibility.


## [v0.11.1] - 2021-08-24
### Changed
- Rather than bootstrapping the training dataset and using the results to
  compute an estimator of the standard deviation, the same training dataset is
  trained on all ten times, and the mean of these along with a confidence
  interval is outputted.

### Fixed
- Updated the model metadata fetching to the new HTML structure of the
  HuggingFace Hub.
- A random seed is now set for all libraries, via the `transformers.set_seed`
  function.
- Always update the list of all the benchmarks when calling the
  `Benchmark.benchmark` method, to allow for possibility of setting new
  benchmark parameters after initialisation.


## [v0.11.0] - 2021-08-23
### Added
- The subjective/objective part of the `TwitterSent` and `Europarl2` datasets
  have now been added as binary classification tasks, called `TwitterSubj` and
  `EuroparlSubj`, respectively. These can now be benchmarked with the
  `Benchmark` class and the CLI using the `twitter-subj` and `europarl-subj`
  names, respectively.
- Added an abstract `BinaryClassificationBenchmark`, to streamline the binary
  classification benchmark datasets, which now includes the `DKHate`,
  `TwitterSubj` and `EuroparlSubj` datasets.


## [v0.10.1] - 2021-08-20
### Fixed
- Now catches `IndexError` during training.


## [v0.10.0] - 2021-08-20
### Fixed
- Properly filters by languages now via the `language` argument in the CLI and
  the `Benchmark` class. As HuggingFace Hub does not have a keyword for
  language, a search for language also means that any other non-language tag
  with that name also shows up in the results. These are now manually removed.
  This means it takes a few more seconds to compile the model list, but it will
  at least be accurate.
- In case `model_max_length` has not been set in a model configuration, it
  defaults to the value of `max_position_embeddings`. This fixes a problem with
  some models not being able to be trained on datasets whose texts were too
  long.
- Now handles the case where a non-classification model, such as a seq-to-seq
  model, are being benchmarked on a classification dataset.

### Added
- All the benchmark classes and `Benchmark` now has a `benchmark` method, which
  does the same as the `__call__` method. This is primarily so that it shows up
  in the Sphinx documentation.
- Added the default `LABEL_0` and `LABEL_1` label synonyms for `NOT` and `OFF`
  in the `DKHate` benchmark.
- Added the possibility of benchmarking randomly initialised RoBERTa models,
  using the model IDs `random-roberta-sequence-clf` and
  `random-roberta-token-clf`.


## [v0.9.0] - 2021-08-19
### Added
- Added the separate `nb` (Norwegian Bokmål) and `nn` (Norwegian Nynorsk)
  language tags, on top of the general `no` (Norwegian).
- Added more multilingual models.

### Fixed
- SpaCy models was evaluated wrongly on the `dane-no-misc` dataset, as their
  `MISC` predictions was not replaced with `O` tags.
- When evaluating models finetuned for token classification on a text
  classification task, a `ValueError` was raised, rather than an
  `InvalidBenchmark` exception.
- If none of the model's labels are among the dataset's labels, and are not
  even synonyms of them, then raise an `InvalidBenchmark`. This prevents things
  like evaluating a finetuned sentiment model on a NER task.
- When `evaluate_train` was `True`, this previously evaluated the test set
  instead.

### Changed
- Changed `Benchmark` API. Now the constructor and the `__call__` method have
  the same arguments, except the `model_id` and `dataset` in `__call__`, where
  the constructor sets the default values and the `__call__` method can change
  these to specific cases.
- Changed the benchmarking order. Now benchmarks all datasets for a model,
  before moving on to the next model
- Renamed the `multilabel` argument to the more descriptive `two_labels`.
- Updated docstrings to be more accurate.
- Early stopping patience is now set to `2 + 250 // len(train)`, so that
  smaller datasets can enjoy a bit more patience, but if the dataset contains
  at least 250 samples then it will remain at the current 2 patience.

### Removed
- Removed `learning_rate`, `batch_size`, `warmup_steps` and `num_finetunings`
  arguments from the benchmarks. These are now fixed to 2e-5, 32, 25% of the
  training dataset and 10, respectively. Note that the batch size will still
  automatically decrease if the GPU runs out of memory.


## [v0.8.0] - 2021-08-18
### Changed
- Models are now being trained for much longer, but with an early stopping
  callback with patience 2. This will enable a more uniform comparison between
  models that require a different number of finetuning epochs.

### Fixed
- There was a bug when evaluating a finetuned PyTorch model on a sequence
  classification task, if the model had only been trained on a proper subset of
  the labels present in the dataset.

### Removed
- All individual benchmarks have been removed from `__init__.py`. They can
  still be imported using their individual modules, for instance
  `from scandeval.dane import DaneBenchmark`, but the idea is to use the
  general `Benchmark` class instead.


## [v0.7.0] - 2021-08-17
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
- Features Danish sentiment, hate speech detection and named entity
  recognition datasets for benchmarking
