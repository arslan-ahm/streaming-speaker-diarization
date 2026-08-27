# Reproducibility

Every number in [RESULTS.md](RESULTS.md) and the README comes from a CSV under
`results/tables/`, produced by the commands below on the machine described below.
Nothing is copied from a paper, and nothing is estimated.

---

## Environment

| item | value |
|---|---|
| OS | Windows 10 Pro 19045 |
| CPU | 4 cores, shared with other jobs during these runs |
| RAM | 16 GB |
| GPU | none — every result here is CPU-only |
| Python | 3.12.13 (pinned via `.python-version`) |
| torch | 2.13.0+cpu |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| package manager | `uv` 0.11.9 |

**SciPy is not a dependency.** The copy originally present in this environment
imported without `scipy.__config__` and crashed; pandas was also broken (its
package directory had no `__init__.py`). Rather than depend on a fragile install,
the Hungarian solver, the Wilcoxon signed-rank test, the bootstrap, Holm-Bonferroni,
agglomerative clustering, spectral clustering and k-means++ are all hand-rolled on
NumPy. Where SciPy happens to be importable the test suite uses it as an *optional*
independent cross-check and skips cleanly when it is not.

Thread budget: every entry point calls `limit_threads(2)`, which sets
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` and `torch.set_num_threads(2)`. Other agents
were building other projects on the same four cores during these runs, so
wall-clock figures below are upper bounds and the efficiency table's median/IQR
are the numbers to trust.

## Install

```bash
uv sync                     # or: uv pip install -e ".[dev,notebooks]"
uv run pytest tests -m "not slow"
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so a fresh checkout can
run the suite with no install step.

---

## The full pipeline, in order

Total local compute for everything below is about **75 minutes** on this machine.
No single command exceeds 5 minutes except a training run (~4.5 min).

### 1. Train the embedders

```bash
# the three seeds used for every headline number
for s in 0 1 2; do
  uv run python scripts/train.py --config configs/base.yaml --seed $s
done

# the non-causal ablation needs its own architecture, hence its own checkpoints
for s in 0 1 2; do
  uv run python scripts/train.py --config configs/ablation_noncausal.yaml --seed $s \
    --checkpoint checkpoints/noncausal_seed$s.pt
done
```

Measured: **262 s** per run for 900 steps (0.259 s/step), 52,144 parameters.
Writes `results/runs/<name>_seed<s>/{config.yaml,history.jsonl,summary.json}` and a
checkpoint under `checkpoints/` (not committed — regenerable from the seed in under
five minutes, so committing 200 KB of weights would add nothing).

### 2. Hyperparameter selection — dev split, seed 0 only

```bash
uv run python scripts/tune_thresholds.py --config configs/base.yaml
uv run python scripts/tune_momentum.py   --config configs/base.yaml
```

Writes `results/tables/threshold_tuning_*.csv` and `momentum_tuning.csv`. The
chosen values are frozen into `configs/base.yaml` by hand, so seeds 1 and 2 are
held out with respect to hyperparameter choice. See RESULTS.md §"What was tuned"
for the selection rule and the disclosure about *when* the momentum was tuned.

### 3. The headline experiment — chunked, resumable

```bash
for s in 0 1 2; do
  for b in 0 125 250 500 1000 2000 4000 8000 16000; do
    uv run python scripts/latency_sweep.py --budget-ms $b --seed $s
  done
  uv run python scripts/latency_sweep.py --offline --seed $s
done
uv run python scripts/latency_sweep.py --summarise
```

One invocation per `(budget, seed)` cell, appending per-recording rows to
`results/tables/latency_sweep.csv` before the next begins. A cell already present
in the CSV is skipped, so an interrupted sweep resumes rather than restarts. This
matters: the sweep is 33 cells and a crash partway through would otherwise cost
the whole curve.

### 4. Comparison, ablations, calibration, efficiency

```bash
uv run python scripts/run_experiments.py --stage comparison  --seeds 0,1,2
uv run python scripts/run_experiments.py --stage ablations   --seeds 0,1,2
uv run python scripts/run_experiments.py --stage calibration --seeds 0,1,2
uv run python scripts/run_experiments.py --stage efficiency  --seeds 0
```

### 5. Figures and notebooks

```bash
uv run python scripts/make_figures.py
uv run python scripts/make_notebooks.py --execute
```

Figures read only the committed CSVs — no figure recomputes a number, so a figure
cannot disagree with a table.

### Smoke path (well under a minute, used by CI)

```bash
uv run python scripts/train.py --config configs/smoke.yaml --seed 0
uv run python scripts/run_experiments.py --stage comparison --config configs/smoke.yaml --seeds 0
```

---

## Determinism evidence

The determinism claims are asserted by the test suite rather than only documented,
because a documented claim rots and a test does not.

| property | evidence |
|---|---|
| Same seed, same generated features | `tests/test_data.py::TestDeterminism` — `np.array_equal` on the feature matrices |
| Same seed, same trained weights | `test_engine.py::test_same_seed_gives_identical_weights` — `torch.equal` on every parameter |
| Checkpoint round-trip | `test_engine.py::TestCheckpoint` — identical embeddings after save/load |
| Repeated inference | `test_causality.py::test_repeated_runs_are_bit_identical` — max abs difference **0.0** on per-window confidences |
| Offline clustering | `test_offline.py::test_is_deterministic` — seeded k-means++ restarts give identical labels |

The measured max-absolute-difference across a repeated end-to-end run of the same
command is **0.0** for per-window labels, confidences and emission frames. The
`--summarise` and figure steps are pure functions of the CSVs.

### The causality guarantee

This one is worth separating out, because it is a *guarantee* rather than a
statistic. Take a recording, replace every frame after `t0` with noise, and re-run:

| variant | windows checked | max abs difference in embeddings |
|---|---|---|
| `causal: true` (shipped) | 48 | **0.0** |
| `causal: false` (ablation) | 48 | 5.7e-3 |

Reproduce with `pytest tests/test_causality.py -v`, or interactively in
`notebooks/01_data_and_the_core_idea.ipynb`. The same property is asserted at the
VAD level and at the level of the diarizer's emitted decisions.

---

## Known non-determinism and its scope

* **Wall-clock times vary** with machine load. Other agents were compiling and
  training on the same four cores. This is why the efficiency table reports median
  and IQR over >= 25 repeats after >= 8 warm-up iterations rather than a single
  timing, and why the *emission latency* claims are stated in audio time (a
  property of the algorithm) rather than compute time.
* **`tracemalloc` peak memory** counts Python-side allocations only. It does not
  see torch's allocator arena or NumPy buffers allocated inside C extensions, so
  it undercounts absolute RSS. It is used only for relative comparison between two
  pipelines allocating through the same paths.
* **Test recordings depend on the seed** as well as the model. That is deliberate
  — the seed study varies the whole pipeline, which is the honest run-to-run scale
  — but it means a per-recording paired test is only valid *within* one seed, and
  every paired test in this repository is run that way.
