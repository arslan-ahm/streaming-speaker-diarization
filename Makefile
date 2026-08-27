# Convenience targets. Everything here is a thin wrapper over the scripts, which
# are themselves thin wrappers over src/streamdiar/pipelines -- so nothing in this
# file is load-bearing and the pipeline can always be driven directly.

PY      ?= uv run python
SEEDS   ?= 0,1,2
CONFIG  ?= configs/base.yaml
BUDGETS ?= 0 125 250 500 1000 2000 4000 8000 16000

.PHONY: help setup lint test test-all smoke train tune sweep experiments figures notebooks all clean

help:
	@echo "setup       - uv sync (Python 3.12, CPU torch)"
	@echo "lint        - ruff check src scripts tests"
	@echo "test        - pytest -m 'not slow'   (~2 min)"
	@echo "test-all    - pytest, including slow end-to-end smoke tests"
	@echo "smoke       - train + compare at tiny scale, asserting a populated table"
	@echo "train       - train the 3 base seeds and the 3 non-causal seeds"
	@echo "tune        - dev-split threshold and momentum selection (seed 0 only)"
	@echo "sweep       - the headline DER-vs-latency sweep, chunked per budget"
	@echo "experiments - comparison, ablations, calibration, efficiency"
	@echo "figures     - render every figure from the committed CSVs"
	@echo "notebooks   - generate and execute notebooks 01-04"
	@echo "all         - train, tune, sweep, experiments, figures, notebooks"

setup:
	uv sync --extra dev --extra notebooks

lint:
	uv run ruff check src scripts tests

test:
	uv run pytest tests -m "not slow" -q

test-all:
	uv run pytest tests -q

smoke:
	$(PY) scripts/train.py --config configs/smoke.yaml --seed 0 --quiet
	$(PY) scripts/run_experiments.py --stage comparison --config configs/smoke.yaml --seeds 0

train:
	@for s in 0 1 2; do $(PY) scripts/train.py --config $(CONFIG) --seed $$s; done
	@for s in 0 1 2; do $(PY) scripts/train.py --config configs/ablation_noncausal.yaml \
		--seed $$s --checkpoint checkpoints/noncausal_seed$$s.pt; done

tune:
	$(PY) scripts/tune_thresholds.py --config $(CONFIG)
	$(PY) scripts/tune_momentum.py   --config $(CONFIG)

# One invocation per (budget, seed) cell so an interrupted sweep resumes rather
# than restarts. Cells already present in the CSV are skipped.
sweep:
	@for s in 0 1 2; do \
		for b in $(BUDGETS); do \
			$(PY) scripts/latency_sweep.py --budget-ms $$b --seed $$s; \
		done; \
		$(PY) scripts/latency_sweep.py --offline --seed $$s; \
	done
	$(PY) scripts/latency_sweep.py --summarise

experiments:
	$(PY) scripts/run_experiments.py --stage comparison  --seeds $(SEEDS)
	$(PY) scripts/run_experiments.py --stage ablations   --seeds $(SEEDS)
	$(PY) scripts/run_experiments.py --stage calibration --seeds $(SEEDS)
	$(PY) scripts/run_experiments.py --stage efficiency  --seeds 0

figures:
	$(PY) scripts/make_figures.py

notebooks:
	$(PY) scripts/make_notebooks.py --execute \
		--only 01_data_and_the_core_idea,02_latency_tradeoff,03_ablations,04_calibration_and_abstention
	$(PY) scripts/make_notebooks.py --only 05_colab_full_scale

all: train tune sweep experiments figures notebooks

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
