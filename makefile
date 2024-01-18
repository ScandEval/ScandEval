# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help

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
	@$(MAKE) --quiet setup-git
	@echo "Installed the 'ScandEval' project."

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
	@if [ ! "$(shell poetry --version)" = "Poetry (version 1.5.1)" ]; then \
		python3 -m pip uninstall -y poetry poetry-core poetry-plugin-export; \
		pipx install --force poetry==1.5.1; \
		echo "Installed Poetry."; \
	fi

setup-poetry:
	@poetry env use python3.10 && poetry install

setup-environment-variables:
	@poetry run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@poetry run python src/scripts/fix_dot_env_file.py --non-interactive

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}
	@poetry run pre-commit install

docs:  ## Generate documentation
	@poetry run pdoc --docformat google src/scandeval -o docs
	@echo "Saved documentation."

view-docs:  ## View documentation
	@echo "Viewing API documentation..."
	@uname=$$(uname); \
		case $${uname} in \
			(*Linux*) openCmd='xdg-open'; ;; \
			(*Darwin*) openCmd='open'; ;; \
			(*CYGWIN*) openCmd='cygstart'; ;; \
			(*) echo 'Error: Unsupported platform: $${uname}'; exit 2; ;; \
		esac; \
		"$${openCmd}" docs/{{ cookiecutter.project_name }}.html

test:  ## Run tests
	@poetry run pytest && poetry run readme-cov && rm .coverage*

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
