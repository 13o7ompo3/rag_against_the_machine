.PHONY: install run debug clean lint lint-strict

CACHE_DIR := $(HOME)/goinfre/

export HF_HOME := $(CACHE_DIR)/.cache/hf
export UV_PROJECT_ENVIRONMENT := $(CACHE_DIR)/.venv
export UV_CACHE_DIR := $(CACHE_DIR)/.cache/uv

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	uv run flake8 src/
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src/

lint-strict:
	uv run flake8 src/
	uv run mypy --strict src/