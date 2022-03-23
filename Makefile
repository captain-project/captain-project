.PHONY: build local-install activate-env

build:
	python -m build

local-install:
	pip install -e .

activate-env:
	conda activate captain
