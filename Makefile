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

test:
	uv run python3 -m src search_dataset data/datasets/public/AnsweredQuestions/dataset_code_public.json 10 . $(ARGS) >/dev/null 2>&1
	uv run python3 -m src search_dataset data/datasets/public/AnsweredQuestions/dataset_docs_public.json 10 . $(ARGS) >/dev/null 2>&1
	uv run python3 -m src search_dataset data/datasets/private/AnsweredQuestions/dataset_code_private.json 10 . $(ARGS) >/dev/null 2>&1
	uv run python3 -m src search_dataset data/datasets/private/AnsweredQuestions/dataset_docs_private.json 10 . $(ARGS) >/dev/null 2>&1
	@echo -e '\n'
	uv run python3 -m src evaluate dataset_code_public.json data/datasets/public/AnsweredQuestions/dataset_code_public.json
	@echo -e '\n'
	uv run python3 -m src evaluate dataset_docs_public.json data/datasets/public/AnsweredQuestions/dataset_docs_public.json
	@echo -e '\n'
	uv run python3 -m src evaluate dataset_code_private.json data/datasets/private/AnsweredQuestions/dataset_code_private.json
	@echo -e '\n'
	uv run python3 -m src evaluate dataset_docs_private.json data/datasets/private/AnsweredQuestions/dataset_docs_private.json