.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:
export ENV_DIR=$( poetry env list --full-path | grep Activated | cut -d' ' -f1 )"

activate:
	@echo "Activating virtual environment..."
	@poetry shell
	@source "$(ENV_DIR)/bin/activate"

install:
	@echo "Installing..."
	@git init
	@if [ "Type `gpg --list-keys` to see your key IDs" != "Type `gpg --list-keys` to see your key IDs" ]; then\
		git config commit.gpgsign true;\
		git config user.signingkey "Type `gpg --list-keys` to see your key IDs";\
	fi
	@git config user.email "saattrupdan@gmail.com"
	@git config user.name "Dan Saattrup Nielsen"
	@poetry install
	@poetry run pre-commit install

remove-env:
	@poetry env remove python3
	@echo "Removed virtual environment."

view-docs:
	@echo "Viewing API documentation..."
	@pdoc src/{{cookiecutter.project_name}}

docs:
	@pdoc src -o docs
	@echo "Saved documentation."

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@echo "Cleaned repository."
