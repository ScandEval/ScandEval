# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Documentation section and link to `README`
- The `Benchmark` class and the CLI now accepts a `batch_size` argument

### Changed
- `Benchmark` arguments `languages`, `tasks`, `model_ids` and `datasets` have
  been renamed to `language`, `task`, `model_id` and `dataset`, to keep it
  consistent with the CLI.
- `batch_size` arguments can now only be among 1, 2, 4, 8, 16 and 32, and the
  corresponding gradient accumulation will be set to 32, 16, 8, 4, 2 and 1,
  respectively. This is to ensure that all finetuning is done using the same
  effective batch size, to ensure fair comparisons.
- Batch sizes are automatically halved if the GPU runs out of memory, with
  gradient accumulation correspondingly doubles.

### Fixed
- `README` typos fixed, and image renders correctly


## [v0.1.0] - 2021-08-05
### Added
- First beta release
- Features Danish sentiment, hate speech detection and named entitity
  recognition datasets for benchmarking
