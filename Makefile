.PHONY: help setup setup-dev test test-fast lint format run analyze controls targets clean

PY ?= python

help:
	@echo "setup       Install the package and its dependencies (correct order)"
	@echo "setup-dev   setup + pytest/ruff"
	@echo "test        Run the full test suite"
	@echo "test-fast   Skip tests that need torch/DeepPurpose"
	@echo "lint        ruff check"
	@echo "format      ruff format"
	@echo "controls    Verify the control structure set"
	@echo "targets     Fetch target sequences only"
	@echo "run         Screen data/example_ingredients.csv"
	@echo "analyze     Post-screen diagnostics (cross-target + MW baseline)"
	@echo "clean       Remove outputs, caches and downloaded checkpoints"

# Install order matters. DeepPurpose must go in with --no-deps: its declared
# requirements pull ax-platform and dgllife, which this project never uses and which
# inflate the environment from ~82 to ~244 packages. descriptastorus is a hard import
# of DeepPurpose.utils, is not on PyPI, and is missing from DeepPurpose's own
# requirements.txt -- hence the explicit git install.
setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install --no-deps DeepPurpose==0.1.5
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .
	@echo ""
	@echo "Verifying the install:"
	@$(PY) -c "import dti_screen, DeepPurpose; print('  dti_screen', dti_screen.__version__, '+ DeepPurpose OK')"

setup-dev: setup
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/

test-fast:
	$(PY) -m pytest tests/ -m "not needs_deeppurpose"

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

controls:
	$(PY) -m dti_screen.cli controls

targets:
	$(PY) -m dti_screen.cli targets --config config/default.yaml

run:
	$(PY) -m dti_screen.cli run --config config/default.yaml

analyze:
	$(PY) -m dti_screen.cli analyze --indir predictions

clean:
	rm -rf predictions dp_work .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
