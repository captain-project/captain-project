.PHONY: build local-install activate-env

build:
	rm -rf dist
	hatch build

publish-test:
	hatch publish -r test

publish:
	hatch publish

local-install:
	pip install -e .

activate-env:
	conda activate captain
