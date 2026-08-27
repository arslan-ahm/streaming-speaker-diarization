![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20CPU-red)
![Tests](https://img.shields.io/badge/tests-649%20passing-brightgreen)
![Params](https://img.shields.io/badge/params-52.1K-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# The Latency Budget Nobody Measured

**Online speaker diarization under a bounded emission delay, with the offline
global-clustering reference implemented in the same codebase as the
infinite-latency asymptote.**

**Efficiency axis: emission latency.** The online system commits a speaker label
**59× sooner** than the offline reference — a measured median emission delay of
**500 ms against 29,500 ms** on 60-second recordings (`results/tables/efficiency.csv`)
— and its clustering cost is flat per window, so it is **5.1× faster to cluster a
4-minute recording** and the gap widens with length. It pays **+0.028 DER**
against the stronger of the two offline baselines, which is 1.0× the run-to-run
noise scale.

> **Result, up front, and it is not the result this project was designed to
> find.** The efficiency and causality claims hold and are measurements. The
> **DER-versus-latency tradeoff curve does not exist at this scale**: across
> budgets from 0 ms to 16,000 ms the DER range is 0.0178, which is **0.32× the
> run-to-run noise scale**. The bounded-lookahead mechanism the project is named
> after is **ablated to nothing** (0.04× the noise scale). An earlier pass *did*
> show a large tradeoff, and that turned out to be the lookahead compensating for
> a badly-tuned centroid update rule — it is retracted in full in
> [docs/RESULTS.md](docs/RESULTS.md#retractions). What survives statistical
> scrutiny is two *other* mechanisms, a speaker-counting difference, and the
> latency measurement itself.

---

## The one-paragraph argument

Every strong diarization system since Sell & Garcia-Romero (2014) extracts
embeddings over sliding windows and then clusters them **globally**. That is why
it works and why it cannot be used live: global clustering sees every embedding
before it assigns any label, so the first label arrives after the last frame. This
repository asks what a fixed emission-delay budget costs. Formally, offline
computes `L_k = f(e_1..e_K)` for every `k`; a bounded-latency system computes
`L_k = f(e_1..e_{k+floor(B/hop)})` and must commit at `end_k + B`, never revising.
Offline is the `B → ∞` limit, so the deliverable is the cost curve `DER(B)` — not a
win over it.

## Causality is a guarantee here, not a statistic

This is the strongest thing in the repository. Take a recording, replace every
frame after `t0` with noise, re-run, and compare the embeddings of every window
that closed at or before `t0`:

| variant | windows checked | max abs difference |
|---|---|---|
| `causal: true` (shipped) | 48 | **0.0** |
| `causal: false` (ablation) | 48 | 6.1e-3 |

Not "close" — bit-identical. Three separate leaks are closed to get there: centred
convolution padding, normalisation that reduces over time, and per-recording
feature standardisation (the common one, and invisible because it *improves*
offline numbers). The non-causal variant is **required to fail** the same test, so
the clean result is not vacuous.
→ `tests/test_causality.py`, asserted at the conv-stack, VAD and decision levels.

And causality is nearly free: the non-causal ablation scores DER 0.2880 against
the causal 0.3020, a difference of 0.86× the noise scale.

---

## Results

Generated data, 24 test recordings × 60 s, 2–5 speakers, 8.2% of speech frames
overlapped (reference-speaker-time miss floor 0.0749), 3 seeds. Mean true speaker
count 3.74. DER scored with **collar 0 and overlap included**, which is
stricter than the NIST convention. Full tables in
[docs/RESULTS.md](docs/RESULTS.md).

### The headline: DER versus latency budget

`results/tables/latency_curve.csv`, `results/figures/der_vs_latency.png`

| budget (ms) | 0 | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|---|---|---|---|
| **DER** | 0.3160 | 0.3160 | 0.3059 | 0.3020 | **0.3013** | 0.3049 | 0.3191 | 0.3159 | 0.3160 |
| noise scale `√2·σ_seed` | 0.056 | 0.056 | 0.037 | 0.027 | 0.038 | 0.013 | 0.025 | 0.020 | 0.019 |
| windows averaged per decision | 1.00 | 1.00 | 1.96 | 2.88 | 4.53 | 7.16 | 10.39 | 12.96 | 13.95 |

**Range across the whole sweep: 0.0178. Largest noise scale: 0.0558. Ratio 0.32.**
The curve is flat inside noise. The shallow minimum at 500–1000 ms and the rise
beyond 2000 ms are *suggestive of* a real optimum — a lookahead long enough to
denoise the query but short enough not to average across turn boundaries — but at
0.32× the noise scale this repository cannot claim it, and does not.

0 ms and 125 ms are identical by construction: `floor(B/hop)` with `hop = 250 ms`,
so anything under one hop buys no lookahead at all. Both points are in the sweep to
make that step visible rather than assumed away.

### Online against the reference approach

`results/tables/method_summary.csv`

| system | DER | `√2·σ_seed` | confusion | JER | speakers found (true 3.74) | median emission delay |
|---|---|---|---|---|---|---|
| **online** (B = 500 ms) | 0.3020 | 0.027 | 0.1972 | 0.510 | 4.33 | **500 ms** |
| naive online (B = 0) | 0.2916 | 0.041 | 0.1867 | 0.496 | 3.96 | **0 ms** |
| offline AHC | 0.3303 | 0.043 | 0.2254 | 0.584 | 3.14 | 29,634 ms |
| offline spectral | 0.2738 | 0.028 | 0.1689 | 0.500 | 3.22 | 29,634 ms |

Read that table carefully, because it does not say what "offline beats online"
would say:

* **The online system beats one offline baseline and loses to the other.** The
  spread *between* the two standard offline clustering algorithms (0.057) is twice
  the gap from online to the better of them (0.028). Which clustering algorithm you
  pick matters more than whether you are online.
* **Miss and false alarm are identical for every non-oracle row** (0.0809 / 0.0239)
  because all systems share one causal VAD and one window grid. Every between-method
  difference is *speaker confusion*, and the tables say so.
* **The overlap floor is 0.0749 and is shared by all of them** — every system here
  emits at most one speaker per window, so the overlap fraction is an irreducible
  miss. Subtract it before comparing to a published DER.

### What survives statistical scrutiny

Paired Wilcoxon + bootstrap CI (2000 resamples) over 24 recordings, Holm-corrected
across the five-metric family, each delta then placed against `√2·σ_seed` from the
3-seed study. `results/tables/statistical_tests.csv`. The paired tests run **within
one seed** (the test recordings are generated per seed, so pooling would break the
pairing), which is why a paired delta differs from the 3-seed mean gap.

| claim | verdict |
|---|---|
| Online over-counts speakers, every offline variant under-counts (+0.67 vs −0.58 to −0.92) | **survives** (ratio 3.2–13.9, p_adj ≤ 0.01) |
| Removing new-speaker spawning costs +0.231 DER | **survives** (ratio 8.70) |
| Removing centroid memory costs +0.110 DER | **survives** (ratio 4.15) |
| Offline spectral beats online on DER (0.028 on 3-seed means, 0.058 paired on seed 0) | not significant (p_adj 0.40, ratio 2.06) |
| Online beats offline AHC on DER (0.028 / 0.035) | inside noise (ratio 0.81) |
| Naive online beats online on DER (0.010 / 0.040) | inside noise (ratio 0.98) |
| **The bounded-lookahead mechanism contributes anything** | **inside noise (ratio 0.04)** |
| Oracle speaker count helps DER | inside noise (ratio 0.16) |
| Oracle VAD helps DER | inside noise (ratio 0.10) |

The only DER-level mechanisms that survive are **spawning** and **centroid
memory** — neither of which is the latency mechanism. The one thing that survives
strongly across systems is a *speaker-counting* difference, not an accuracy one.

### Ablations, one switch each

`results/tables/ablation_summary.csv`

| ablation | DER (3 seeds) | cost of removal (paired, seed 0) | ratio to noise | verdict |
|---|---|---|---|---|
| full | 0.3020 | — | — | — |
| `bounded_window: false` | 0.3160 | +0.0025 | 0.04 | **inside noise** |
| `spawn_enabled: false` | 0.5245 | +0.2314 | 8.70 | **survives** |
| `centroid_momentum: 1.0` | 0.4366 | +0.1104 | 4.15 | **survives** |
| `causal: false` | 0.2880 | −0.0228 | 0.86 | inside noise |

The DER column is the mean over 3 seeds; the delta column is the *paired* delta
over the 24 recordings of seed 0, which is what the test is computed on. The two
differ, and both are reported rather than one being quietly substituted for the
other.

`bounded_window: false` really does disable the mechanism — the windows-averaged-
per-decision diagnostic drops from 2.88 to exactly 1.00. It is off, and it changes
nothing measurable.

### Calibration and abstention

`results/tables/calibration_summary.csv`. A diarizer that knows which turns it got
wrong can defer them to an offline pass, so this is the practical payoff of the
online design — and it is where the proposal loses to its own baseline.

| system | turn accuracy | mean confidence | ECE raw → after dev-fitted T | error-detection AUROC | error at 100% / 80% / 50% coverage |
|---|---|---|---|---|---|
| online | 0.7964 | 0.6252 | 0.1779 → **0.0835** | 0.6754 | 0.204 / 0.154 / 0.128 |
| naive online | 0.8052 | 0.6389 | 0.1718 → 0.0943 | **0.7392** | 0.195 / 0.130 / 0.092 |
| offline AHC | 0.7584 | 0.8494 | **0.0980** → 0.1432 | 0.6348 | 0.242 / 0.181 / 0.176 |

* Both online systems are materially **under**-confident (−0.17); the offline one is
  **over**-confident (+0.09). Opposite failure modes, opposite fixes.
* Deferring the least-confident 20% of turns cuts the online error rate from 0.204
  to 0.154, a 25% relative reduction. Useful — and far from the oracle floor
  (AURC 0.149 vs 0.023).
* **Negative results, stated:** the naive baseline's confidence is a *better* error
  detector than the proposed system's (0.739 vs 0.675), and temperature scaling
  *degrades* the offline system's ECE (0.098 → 0.143) because its post-hoc
  cluster-similarity confidence is already near-calibrated and the fit sharpens it
  past the optimum.

### Efficiency, measured with warm-up 8 and 25 repeats

`results/tables/efficiency.csv`, `results/tables/scaling.csv`

| system | median emission delay | p95 | total wall-clock / 60 s audio | RTF | clustering at 240 s |
|---|---|---|---|---|---|
| online (B = 500 ms) | **500 ms** | 500 ms | 103.0 ms (IQR 10.4) | 0.0017 | **141.8 ms** |
| naive online | **0 ms** | 0 ms | 88.6 ms (IQR 24.2) | 0.0015 | — |
| offline AHC | 29,500 ms | 56,200 ms | 82.2 ms (IQR 12.3) | 0.0014 | 728.9 ms |
| offline spectral | 29,500 ms | 56,200 ms | 88.7 ms (IQR 11.5) | 0.0015 | 525.8 ms |

* **Emission latency: 59.0× lower** (500 ms vs 29,500 ms). This is the headline
  efficiency number and it is an algorithmic property, not a machine property.
* **Compute is not the win at 60 s** — online is *1.25× slower* in total wall-clock
  than offline AHC, because the tracker runs a decision per window while AHC runs
  one batched matrix job. Reported rather than hidden.
* **Compute becomes the win at length.** Per-window clustering cost is flat for
  online (0.191 → 0.156 ms/window from 58 to 912 windows) and rises 10.9× for AHC
  (0.074 → 0.799). At 240 s of audio online clusters **5.1× faster**, and the ratio
  grows without bound.

---

## Install and 60-second quickstart

```bash
uv sync                                    # Python 3.12, CPU torch, no scipy needed
uv run pytest tests -m "not slow"          # 649 tests, ~2 min

# tiny end-to-end run: trains, diarizes, prints a populated results table
uv run python scripts/train.py --config configs/smoke.yaml --seed 0
uv run python scripts/run_experiments.py --stage comparison --config configs/smoke.yaml --seeds 0
```

Full reproduction (about 75 minutes of CPU) is `make all`, or step by step in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

No network access, no API keys, no dataset downloads. The shipped results come
from a procedural generator whose diarization reference is **exact by
construction** — turns are sampled first and features rendered from them — which
is the point: AMI/VoxConverse references carry tens of milliseconds of human
boundary error, the same order as the differences this sweep tries to resolve. An
optional WAV+RTTM path exists (`src/streamdiar/data/real.py`) and is never on the
critical path.

## Repo map

```
src/streamdiar/
  config.py            dataclasses + YAML `_base_` inheritance + typed --set overrides
  data/generator.py    procedural conversations; reference is generative truth
  data/real.py         optional WAV+RTTM loader, hand-rolled log-mel front end
  models/embedder.py   causal dilated-conv embedder, O(1)-per-window pooling
  models/vad.py        strictly causal energy VAD, threshold fitted on dev
  models/online.py     the bounded-latency tracker + naive online baseline
  models/offline.py    AHC and spectral reference baselines, hand-rolled
  metrics/assignment.py hand-rolled Hungarian (validated against brute force)
  metrics/der.py       md-eval DER, miss/FA/confusion, JER, collar, overlap switches
  metrics/calibration.py ECE/ACE/MCE/Brier/NLL, risk-coverage, AURC, AUROC
  metrics/stats.py     exact Wilcoxon, bootstrap, Holm, the √2·σ noise gate
  engine/              training, checkpointing, inference + scoring
  pipelines/           the experiments; scripts/ are thin wrappers
configs/               base + smoke + one per method and per ablation
notebooks/             01 data & causality · 02 the tradeoff · 03 ablations
                       04 calibration · 05 Colab full-scale (not executed here)
results/tables/        every number in every document
results/figures/       every figure, rendered only from those CSVs
```

## Limitations

* **Single-label output.** No system here emits overlapping speakers, so the
  0.0749 overlap fraction is an irreducible miss for all of them.
* **Small scale.** 24 test recordings × 60 s, 3 seeds, a 52K-parameter embedder on
  4 CPU cores. The `√2·σ_seed` noise scale of ~0.03–0.06 DER is what limits every
  claim above, and a larger study would resolve the shallow curve minimum this one
  cannot.
* **Generated data.** Exact reference, but its own generative model. The real-data
  path is implemented and untested at scale.
* **Unit of analysis.** The paired tests are over recordings within one seed, so
  they condition on one trained model per method. Method-level claims rest on the
  3-seed noise scale, not on the p-values.

## Citations

Sell & Garcia-Romero, *Speaker Diarization with PLDA i-Vector Scoring*, SLT 2014 ·
Wang et al., *Speaker Diarization with LSTM*, ICASSP 2018 ·
Snyder et al., *X-Vectors*, ICASSP 2018 ·
Wan et al., *Generalized End-to-End Loss for Speaker Verification*, ICASSP 2018 ·
Bai et al., *An Empirical Evaluation of Generic Convolutional and Recurrent
Networks*, 2018 ·
Zhang et al., *Fully Supervised Speaker Diarization* (UIS-RNN), ICASSP 2019 ·
Coria et al., *Overlap-Aware Low-Latency Online Speaker Diarization*, 2021 ·
Ryant et al., *The Second DIHARD Challenge*, Interspeech 2019 ·
Ng, Jordan & Weiss, *On Spectral Clustering*, NeurIPS 2001 ·
Jonker & Volgenant, *A Shortest Augmenting Path Algorithm*, Computing 1987 ·
Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 ·
Nixon et al., *Measuring Calibration in Deep Learning*, CVPRW 2019 ·
Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017 ·
Holm, *A Simple Sequentially Rejective Multiple Test Procedure*, 1979.

MIT licensed. Author: Arslan Ahmad.
