# colors
GREEN=\033[0;32m
RED=\033[0;31m
BLUE=\033[0;34m
NC=\033[0m

PYTHON_VERSIONS ?= 3.10.13 3.11.5 3.12.4
PROJECT=$(shell basename $(CURDIR))
PACKAGE_NAME=loomlib

LOG_LEVEL?=ERROR

TEST_ENVS=$(addprefix $(PROJECT)-test-,$(PYTHON_VERSIONS))

# -- development --------------------------------------------------------------

install:
	@echo "$(BLUE)Installing loomlib in editable mode$(NC)"
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	python -m pytest tests/ -v

coverage: test
	coverage run -m pytest
	coverage report
	coverage lcov

# -- documentation -------------------------------------------------------------

docs:
	cd docs && make html
	open docs/_build/html/index.html

# -- packaging -----------------------------------------------------------------

dist: dist-clean
	python -m build

dist-clean:
	rm -rf dist build *.egg-info

publish-test: dist
	twine upload --repository testpypi dist/*

publish: dist
	twine upload dist/*

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

.PHONY: install lint test coverage docs dist dist-clean publish-test publish clean

-include Makefile.mak
