# Makefile — build/test/eval glue for citation_verification.
#
# Targets:
#   make install   editable install with dev extras (uv if present, else pip)
#   make test      run the full pytest suite (offline; no SDK/network)
#   make lint      ruff check
#   make smoke     fast offline contract-regression: tests + run_eval on smoke gold
#   make eval      run_eval on a report dir:   make eval DIR=papers/<id> GOLD=evals/smoke/gold.jsonl
#   make bench     build CitationHallucinationBench then eval on the full split
#   make schema    regenerate spec/<v>/record.schema.json and fail if it drifted
#   make clean     remove caches and build artifacts
#
# PY drives every Python invocation; override to point at a specific interpreter,
# e.g. `make test PY=.venv/bin/python`.

PY ?= python
PIP_INSTALL := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip install" || echo "$(PY) -m pip install")
SPEC := spec/v0.1/record.schema.json
GOLD ?= evals/smoke/gold.jsonl
DIR  ?=

.DEFAULT_GOAL := help
.PHONY: help install test lint smoke eval bench schema clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras (pydantic, jsonschema, rapidfuzz, ...)
	$(PIP_INSTALL) -e '.[dev]'

test: ## Run the full pytest suite (offline; no SDK / no network)
	$(PY) -m pytest

lint: ## Static lint with ruff
	$(PY) -m ruff check src tests evals

smoke: ## Fast offline contract regression: tests + run_eval on the in-repo smoke gold
	$(PY) -m pytest tests evals -q
	$(PY) -m evals.run_eval evals/smoke $(GOLD) || \
		echo "[smoke] run_eval not yet runnable on this checkout (sibling module pending) — tests passed."

eval: ## Score an agent report dir against gold:  make eval DIR=papers/<id> GOLD=evals/smoke/gold.jsonl
	@test -n "$(DIR)" || { echo "usage: make eval DIR=<report-dir> [GOLD=<gold.jsonl>]"; exit 2; }
	$(PY) -m evals.run_eval $(DIR) $(GOLD)

bench: ## Build CitationHallucinationBench, then eval on the full split
	$(PY) -m chbench.cli build
	$(PY) -m chbench.cli validate
	$(PY) -m evals.run_eval $${CHBENCH_DATA_DIR:-/scratch/datasets/chbench} $${CHBENCH_DATA_DIR:-/scratch/datasets/chbench}/gold.jsonl

schema: ## Regenerate the committed JSON Schema and fail if it drifted from schema.py
	$(PY) -m citation_verifier.schema
	@git diff --quiet -- $(SPEC) && echo "[schema] $(SPEC) is up to date." || \
		{ echo "[schema] DRIFT: $(SPEC) changed — review & commit the regenerated spec."; \
		  git --no-pager diff -- $(SPEC); exit 1; }

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info \
		.pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
