# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: notebook docs

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

install-poetry:
	@echo "Installing poetry..."
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -

uninstall-poetry:
	@echo "Uninstalling poetry..."
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - --uninstall

install:
	@echo "Installing..."
	@if [ "$(shell which poetry)" = "" ]; then \
		make install-poetry; \
	fi
	@if [ "$(shell which gpg)" = "" ]; then \
		echo "GPG not installed, so an error will occur. Install GPG on MacOS with "\
			 "`brew install gnupg` or on Ubuntu with `apt install gnupg` and run "\
			 "`make install` again."; \
	fi
	@poetry env use python3
	@poetry run python3 -m src.scripts.fix_dot_env_file
	@git init
	@. .env; \
		git config --local user.name "$${GIT_NAME}"; \
		git config --local user.email "$${GIT_EMAIL}"
	@. .env; \
		if [ "$${GPG_KEY_ID}" = "" ]; then \
			echo "No GPG key ID specified. Skipping GPG signing."; \
			git config --local commit.gpgsign false; \
		else \
			echo "Signing with GPG key ID $${GPG_KEY_ID}..."; \
			git config --local commit.gpgsign true; \
			git config --local user.signingkey "$${GPG_KEY_ID}"; \
		fi
	@poetry install
	@poetry run pre-commit install

remove-env:
	@poetry env remove python3
	@echo "Removed virtual environment."

docs:
	@poetry run pdoc --html src/scandeval -o docs --force
	@echo "Saved documentation."

view-docs:
	@echo "Viewing API documentation..."
	@open docs/scandeval/index.html

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@echo "Cleaned repository."
