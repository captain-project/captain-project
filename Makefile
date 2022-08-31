.PHONY: build local-install activate-env

build:
	rm -rf dist
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload --repository pypi dist/*

local-install:
	pip install -e .

activate-env:
	conda activate captain
