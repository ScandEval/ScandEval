# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: notebook docs help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install-poetry:
	@echo "Installing poetry..."
	@pipx install poetry==1.2.0
	@$(eval include ${HOME}/.poetry/env)

uninstall-poetry:
	@echo "Uninstalling poetry..."
	@pipx uninstall poetry

install: ## Install dependencies
	@echo "Installing..."
	@if [ "$(shell which poetry)" = "" ]; then \
		$(MAKE) install-poetry; \
	fi
	@if [ "$(shell which gpg)" = "" ]; then \
		echo "GPG not installed, so an error will occur. Install GPG on MacOS with "\
			 "`brew install gnupg` or on Ubuntu with `apt install gnupg` and run "\
			 "`make install` again."; \
	fi
	@$(MAKE) setup-poetry
	@$(MAKE) setup-environment-variables
	@$(MAKE) setup-git

setup-poetry:
	@poetry env use python3.10 && poetry install

setup-environment-variables:
	@poetry run python3.10 -m src.scripts.fix_dot_env_file

setup-git:
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}
	@if [ ${GPG_KEY_ID} = "" ]; then \
		echo "No GPG key ID specified. Skipping GPG signing."; \
		git config --local commit.gpgsign false; \
	else \
		echo "Signing with GPG key ID ${GPG_KEY_ID}..."; \
		echo 'If you get the "failed to sign the data" error when committing, try running `export GPG_TTY=$$(tty)`.'; \
		git config --local commit.gpgsign true; \
		git config --local user.signingkey ${GPG_KEY_ID}; \
	fi
	@poetry run pre-commit install

docs:
	@poetry run pdoc --docformat google src/scandeval -o docs
	@echo "Saved documentation."

view-docs:
	@echo "Viewing API documentation..."
	@uname=$$(uname); \
		case $${uname} in \
			(*Linux*) openCmd='xdg-open'; ;; \
			(*Darwin*) openCmd='open'; ;; \
			(*CYGWIN*) openCmd='cygstart'; ;; \
			(*) echo 'Error: Unsupported platform: $${uname}'; exit 2; ;; \
		esac; \
		"$${openCmd}" docs/scandeval.html

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

publish-major: bump-major publish  ## Publish a major version

publish-minor: bump-minor publish  ## Publish a minor version

publish-patch: bump-patch publish  ## Publish a patch version

test:  ## Run tests
	@PYTORCH_ENABLE_MPS_FALLBACK=1 poetry run pytest && readme-cov

tree:  ## Print directory tree
	@tree -a \
		-I ".git" \
		-I ".mypy_cache" \
		-I ".scandeval_cache" \
		-I ".env" \
		-I ".venv" \
		-I "poetry.lock" \
		-I ".ipynb_checkpoints" \
		-I "dist" \
		-I "scandeval_benchmark_results.jsonl" \
		-I ".gitkeep" \
		-I "docs" \
		-I ".coverage*" \
		-I ".DS_Store" \
		-I ".pytest_cache" \
		-I "__pycache__" \
		.
