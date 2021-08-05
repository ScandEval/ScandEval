include .env
export $(shell sed 's/=.*//' .env)

documentation:
	sphinx-apidoc -o docs/source --force scandeval && \
	make -C docs html

release-major:
	pytest -n 4 scandeval && \
	make documentation && \
	python bump_version.py --major && \
	git pull origin main && \
	git push && \
	git checkout main && \
	git merge dev && \
	git push && \
	git push --tags && \
	git checkout dev && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*

release-minor:
	pytest -n 4 scandeval && \
	make documentation && \
	python bump_version.py --minor && \
	git pull origin main && \
	git push && \
	git checkout main && \
	git merge dev && \
	git push && \
	git push --tags && \
	git checkout dev && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*

release-patch:
	pytest -n 4 scandeval && \
	make documentation && \
	python bump_version.py --patch && \
	git pull origin main && \
	git push && \
	git checkout main && \
	git merge dev && \
	git push && \
	git push --tags && \
	git checkout dev && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*
