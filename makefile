# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Create poetry env file if it does not already exist
ifeq (,$(wildcard ${HOME}/.poetry/env))
  $(shell mkdir ${HOME}/.poetry)
  $(shell touch ${HOME}/.poetry/env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Ensure that `pipx` and `poetry` will be able to run, since `pip` and `brew` put these
# in the following folders on Unix systems
export PATH := ${HOME}/.local/bin:/opt/homebrew/bin:$(PATH)

# Prevent DBusErrorResponse during `poetry install`
# (see https://stackoverflow.com/a/75098703 for more information)
export PYTHON_KEYRING_BACKEND := keyring.backends.null.Keyring

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'ScandEval' project..."
	@$(MAKE) --quiet install-brew
	@$(MAKE) --quiet install-pipx
	@$(MAKE) --quiet install-poetry
	@$(MAKE) --quiet setup-poetry
	@$(MAKE) --quiet setup-environment-variables
	@echo "Installed the 'ScandEval' project. If you want to use pre-commit hooks, run 'make install-pre-commit'."

install-brew:
	@if [ $$(uname) = "Darwin" ] && [ "$(shell which brew)" = "" ]; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Installed Homebrew."; \
	fi

install-pipx:
	@if [ "$(shell which pipx)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Darwin*) installCmd='brew install pipx'; ;; \
				(*CYGWIN*) installCmd='py -3 -m pip install --upgrade --user pipx'; ;; \
				(*) installCmd='python3 -m pip install --upgrade --user pipx'; ;; \
			esac; \
			$${installCmd}; \
		pipx ensurepath --force; \
		echo "Installed pipx."; \
	fi

install-poetry:
	@if [ ! "$(shell poetry --version)" = "Poetry (version 1.8.2)" ]; then \
		python3 -m pip uninstall -y poetry poetry-core poetry-plugin-export; \
		pipx install --force poetry==1.8.2; \
		echo "Installed Poetry."; \
	fi

setup-poetry:
	@if [ "$(shell which nvidia-smi)" = "" ]; then \
	    poetry env use python3.10 && poetry install --extras cpu_all; \
	else \
	    poetry env use python3.10 && poetry install --extras all; \
	fi

install-pre-commit:  ## Install pre-commit hooks
	@poetry run pre-commit install

lint:  ## Lint the code
	@poetry run ruff check . --fix

format:  ## Format the code
	@poetry run ruff format .

type-check:  ## Run type checking
	@poetry run mypy . --install-types --non-interactive --ignore-missing-imports --show-error-codes --check-untyped-defs

check: lint format type-check  ## Run all checks

setup-environment-variables:
	@poetry run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@poetry run python src/scripts/fix_dot_env_file.py --non-interactive

test:  ## Run tests
	@if [ "$(shell which nvidia-smi)" != "" ]; then \
		$(MAKE) --quiet test-cuda-vllm; \
	else \
		$(MAKE) --quiet test-cpu; \
	fi
	@$(MAKE) --quiet update-coverage-badge
	@date "+%H:%M:%S ⋅ All done!"

test-cuda-vllm:
	@rm tests_with_cuda_and_vllm.log; \
		date "+%H:%M:%S ⋅ Running tests with CUDA and vLLM..." \
		&& USE_CUDA=1 USE_VLLM=1 \
			poetry run pytest | tee tests_with_cuda_and_vllm.log \
		&& date "+%H:%M:%S ⋅ Finished testing with CUDA and vLLM!"

test-cuda-no-vllm:
	@rm tests_with_cuda_and_no_vllm.log; \
		date "+%H:%M:%S ⋅ Running tests with CUDA and no vLLM..." \
		&& USE_CUDA=1 USE_VLLM=0 \
			poetry run pytest | tee tests_with_cuda_and_no_vllm.log \
		&& date "+%H:%M:%S ⋅ Finished testing with CUDA and no vLLM!"

test-cpu:
	@rm tests_with_cpu.log; \
		date "+%H:%M:%S ⋅ Running tests with CPU..." \
		&& USE_CUDA=0 poetry run pytest | tee tests_with_cpu.log \
		&& date "+%H:%M:%S ⋅ Finished testing with CPU!"

test-fast:  # Run CPU tests without evaluations
	@rm tests_with_cpu_fast.log; \
		date "+%H:%M:%S ⋅ Running fast tests with CPU..." \
		&& USE_CUDA=0 TEST_EVALUATIONS=0 \
			poetry run pytest | tee tests_with_cpu_fast.log \
		&& date "+%H:%M:%S ⋅ Finished fast testing with CPU!"

test-slow:  # Run all tests
	@if [ "$(shell which nvidia-smi)" != "" ]; then \
		TEST_ALL_DATASETS=1 $(MAKE) --quiet test-cuda-vllm; \
		TEST_ALL_DATASETS=1 $(MAKE) --quiet test-cuda-no-vllm; \
	else \
		TEST_ALL_DATASETS=1 $(MAKE) --quiet test-cpu; \
	fi
	@$(MAKE) --quiet update-coverage-badge
	@date "+%H:%M:%S ⋅ All done!"

update-coverage-badge:
	@poetry run readme-cov
	@rm .coverage*
	@date "+%H:%M:%S ⋅ Updated coverage badge!"

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

bump-major:
	@poetry run python -m src.scripts.versioning --major
	@echo "Bumped major version!"

bump-minor:
	@poetry run python -m src.scripts.versioning --minor
	@echo "Bumped minor version!"

bump-patch:
	@poetry run python -m src.scripts.versioning --patch
	@echo "Bumped patch version!"

publish:
	@if [ ${PYPI_API_TOKEN} = "" ]; then \
		echo "No PyPI API token specified in the '.env' file, so cannot publish."; \
	else \
		echo "Publishing to PyPI..."; \
		poetry publish --build --username "__token__" --password ${PYPI_API_TOKEN}; \
	fi
	@echo "Published!"

publish-major: bump-major publish  ## Publish a major version

publish-minor: bump-minor publish  ## Publish a minor version

publish-patch: bump-patch publish  ## Publish a patch version
