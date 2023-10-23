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

export PATH := $(shell pwd)/.local/bin:$(PATH)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'ScandEval' project..."
	@$(MAKE) --quiet install-brew
	@$(MAKE) --quiet install-gpg
	@$(MAKE) --quiet generate-gpg-key
	@$(MAKE) --quiet install-pipx
	@$(MAKE) --quiet install-poetry
	@$(MAKE) --quiet setup-poetry
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git
	@$(MAKE) --quiet add-repo-to-git
	@echo "Installed the 'ScandEval' project."

install-brew:
	@if [ $$(uname) = "Darwin" ] && [ "$(shell which brew)" = "" ]; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Installed Homebrew."; \
	fi

install-gpg:
	@if [ "$(shell which gpg)" = "" ] || [ "$(shell which gpg-agent)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Linux*) distro=$$(lsb_release -i | sed 's/Distributor ID:\t//'); \
					case $${distro} in \
						(*Ubuntu*) installCmd='apt-get update && apt-get install -y gnupg gpg-agent'; ;; \
						(*CentOS*) installCmd='yum install -y gnupg gpg-agent'; ;; \
						(*) echo 'Could not automatically install gnupg for the $${distro} distribution. Please manually install gnupg and try again.'; exit 2; ;; \
					esac; ;; \
				(*Darwin*) installCmd='brew install gnupg pinentry-mac'; ;; \
				(*) echo 'Could not automatically install gnupg. Please manually install gnupg and try again.'; exit 2; ;; \
			esac; \
			"$${installCmd}"; \
		echo "Installed gnupg."; \
	fi

generate-gpg-key:
	@if [ "$(shell gpg --list-secret-keys --keyid-format=long | grep sec | sed -E 's/.*\/([^ ]+).*/\1/')" = "" ]; then \
		echo "Generating a new GPG key - please follow the prompts."; \
		gpg --full-generate-key; \
		echo "Generated a new GPG key. Remember to register it to Github at https://github.com/settings/gpg/new, where you add the key generated by running 'gpg --armor --export <key>'"; \
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
	@if [ ${GPG_KEY_ID} = "" ]; then \
		echo "No GPG key ID specified. Skipping GPG signing."; \
		git config --local commit.gpgsign false; \
	else \
		git config --local commit.gpgsign true; \
		git config --local user.signingkey ${GPG_KEY_ID}; \
		echo "Signed with GPG key ID ${GPG_KEY_ID}."; \
	fi
	@poetry run pre-commit install

add-repo-to-git:
	@export GPG_TTY=$(tty)
	@gpgconf --kill gpg-agent
	@git add .
	@if [ ! "$(shell git status --short)" = "" ]; then \
		git commit --quiet -m "Initial commit"; \
	fi
	@if [ "$(shell git remote)" = "" ]; then \
		git remote add origin git@github.com:alexandrainst/ScandEval.git; \
	fi

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
	@poetry run pytest && poetry run readme-cov

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
