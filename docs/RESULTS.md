# Results

Every number here is traceable to a CSV under `results/tables/`, named beside the
table it appears in. `uv run python scripts/report.py` prints all of them next to
their sources in one pass, which is how they were cross-checked before this
document was written.

**Read the [retractions](#retractions) section.** It is the most informative part
of this document, and the central claim the project was designed around is in it.

---

## Summary

- **Solid, because they are measurements rather than inferences:** the emission
  latency factor (59.0×, 500 ms vs 29,500 ms), the O(1)-per-window clustering
  scaling (5.1× faster than AHC at 240 s of audio, ratio growing with length),
  bit-identical causality under future perturbation, and bit-identical
  reproducibility.
- **Robust statistical results (survive the noise gate):** removing new-speaker
  spawning costs +0.231 DER (8.70× the noise scale); removing centroid memory
  costs +0.110 DER (4.15×); and the online tracker **over**-counts speakers
  (+0.67) while every offline variant **under**-counts (−0.58 to −0.92), at 3.2–13.9×
  the noise scale.
- **Retracted:** the DER-versus-latency tradeoff curve. Across budgets from 0 to
  16,000 ms the DER range is 0.0178 against a noise scale of 0.0558 — a ratio of
  **0.32**. The curve is flat inside noise.
- **Retracted:** the bounded-lookahead micro-cluster mechanism, i.e. the mechanism
  this project is built around. Ablating it changes DER by 0.0025, **0.04× the
  noise scale**, despite the diagnostic confirming it was fully engaged.
- **Not supported:** "offline beats online". Offline AHC is *worse* than the online
  system (0.3303 vs 0.3020) and offline spectral is better (0.2738). The spread
  between the two standard offline algorithms is 2.0× the gap from online to the
  better of them.
- **Negative result about the proposal's calibration:** the naive baseline's
  confidence is a *better* error detector than the proposed system's (AUROC 0.739
  vs 0.675).
- **Good news for the design:** causality is nearly free. The non-causal ablation
  scores 0.2880 against the causal 0.3020, a difference of 0.86× the noise scale.

---

## What was run, and what was not

| claim | status | where |
|---|---|---|
| Emission latency, RTF, clustering wall-clock for four systems | **measured, final** | `efficiency.csv`, `scaling.csv` |
| Clustering cost scaling over 5 recording lengths | **measured** | `scaling.csv` |
| Causality as bit-identity under future perturbation | **proven by construction + asserted in tests** | `tests/test_causality.py` |
| DER vs latency budget, 9 budgets × 3 seeds | **measured — the curve is flat inside noise** | `latency_curve.csv` |
| Online vs 2 offline baselines + 4 oracle diagnostics, 3 seeds | **measured** | `method_summary.csv` |
| Four ablations, one switch each, 3 seeds | **measured** | `ablation_summary.csv` |
| Paired Wilcoxon + bootstrap + Holm on 5 metrics | **measured** | `statistical_tests.csv`, `ablation_tests.csv` |
| Calibration, temperature scaling, abstention | **measured** | `calibration.csv`, `calibration_summary.csv` |
| Seed-variance study, 3 seeds | **measured** | the `*_seed_sd` / `*_noise_scale` columns |
| Sensitivity of the curve to the centroid update rule | **measured** | `latency_sweep_momentum035.csv`, `latency_curve_momentum035.csv` |
| Real-data (AMI / VoxConverse) numbers | **not measured** | loader implemented, never run at scale |
| More than 3 seeds, or >24 test recordings | **not measured** | this is the binding limitation |
| Overlap-aware (multi-label) output | **not implemented** | 0.0749 miss floor for all systems |

---

## 1. The dataset

`results/tables/method_comparison.csv` (per-recording columns).

| property | value |
|---|---|
| test recordings per seed | 24 |
| duration | 60.0 s each |
| speakers per recording | 2–5, mean **3.74** |
| overlapped fraction of speech | **8.2%** |
| reference speaker-time miss floor from overlap | **0.0749** |
| speaker pool | 160 train / 60 test, disjoint |

The reference is exact by construction: turns are sampled first, features are
rendered from them. That is why generated data carries the shipped results — AMI
and VoxConverse references have human boundary error of tens of milliseconds,
which is the same order as the differences the latency sweep tries to resolve.
See [METHOD.md §2](METHOD.md#2-data-exact-ground-truth-by-construction).

---

## 2. The headline experiment: DER versus latency budget

`results/tables/latency_curve.csv` · `results/figures/der_vs_latency.png`

| budget (ms) | DER | `√2·σ_seed` | confusion | JER | speakers found | windows per decision |
|---|---|---|---|---|---|---|
| 0 | 0.3160 | 0.0558 | 0.2111 | 0.5509 | 4.06 | 1.00 |
| 125 | 0.3160 | 0.0558 | 0.2111 | 0.5509 | 4.06 | 1.00 |
| 250 | 0.3059 | 0.0368 | 0.2011 | 0.5359 | 4.13 | 1.96 |
| 500 | 0.3020 | 0.0266 | 0.1972 | 0.5102 | 4.33 | 2.88 |
| **1000** | **0.3013** | 0.0379 | 0.1965 | 0.5072 | 4.56 | 4.53 |
| 2000 | 0.3049 | 0.0134 | 0.2000 | 0.5012 | 4.79 | 7.16 |
| 4000 | 0.3191 | 0.0253 | 0.2142 | 0.5360 | 4.92 | 10.39 |
| 8000 | 0.3159 | 0.0203 | 0.2110 | 0.5265 | 4.82 | 12.96 |
| 16000 | 0.3160 | 0.0194 | 0.2111 | 0.5268 | 4.82 | 13.95 |

Offline reference, same embedder, same VAD, same scoring:

| system | DER | `√2·σ_seed` | median emission delay |
|---|---|---|---|
| offline AHC | 0.3303 | 0.0427 | 29,634 ms |
| offline spectral | 0.2738 | 0.0283 | 29,634 ms |

### The verdict on the curve

- range across all budgets: **0.0178**
- largest noise scale across budgets: **0.0558**
- **ratio: 0.32**

> **The DER-versus-latency tradeoff curve is not measurable at this scale.** The
> shallow minimum at 500–1000 ms and the degradation beyond 2000 ms are consistent
> with a real mechanism — lookahead long enough to average down query noise, short
> enough not to average across turn boundaries, which is also what the
> windows-per-decision and speaker-count columns suggest as the buffer grows from 1
> to 14 windows — but at a third of the run-to-run noise scale, none of it can be
> claimed.

Two structural facts in the table *are* solid because they are properties of the
construction rather than statistics:

1. **0 ms and 125 ms are bit-identical.** `floor(B / hop)` with `hop = 250 ms`
   means anything below one hop buys zero lookahead. Both points are in the sweep
   to make that step visible rather than assumed away.
2. **Speaker count inflates monotonically with the budget**, 4.06 → 4.92 against a
   true 3.74. A larger micro-cluster spans more turn boundaries, its mean matches
   no existing centroid, and the tracker spawns. This is the mechanism by which a
   longer budget *hurts*, and it is legible even though its DER consequence is not.

---

## 3. Online against the reference approach

`results/tables/method_summary.csv` · `results/figures/der_decomposition.png`

| variant | DER | `√2·σ_seed` | miss | FA | confusion | JER | speakers (true 3.74) | median delay | RTF |
|---|---|---|---|---|---|---|---|---|---|
| **online** (B = 500 ms) | 0.3020 | 0.027 | 0.0809 | 0.0239 | 0.1972 | 0.510 | 4.33 | 500 ms | 0.0027 |
| naive online (B = 0) | 0.2916 | 0.041 | 0.0809 | 0.0239 | 0.1867 | 0.496 | 3.96 | 0 ms | 0.0020 |
| offline AHC | 0.3303 | 0.043 | 0.0809 | 0.0239 | 0.2254 | 0.584 | 3.14 | 29,634 ms | 0.0022 |
| offline spectral | 0.2738 | 0.028 | 0.0809 | 0.0239 | 0.1689 | 0.500 | 3.22 | 29,634 ms | 0.0025 |
| online, oracle count | 0.3038 | 0.035 | 0.0809 | 0.0239 | 0.1989 | 0.559 | 3.42 | 500 ms | 0.0032 |
| offline AHC, oracle count | 0.3330 | 0.047 | 0.0809 | 0.0239 | 0.2281 | 0.615 | 2.92 | 29,634 ms | 0.0024 |
| online, oracle VAD | 0.2950 | 0.041 | 0.0850 | 0.0118 | 0.1982 | 0.506 | 4.33 | 500 ms | 0.0026 |
| offline AHC, oracle VAD | 0.3176 | 0.040 | 0.0850 | 0.0118 | 0.2208 | 0.570 | 3.21 | 29,627 ms | 0.0020 |

The oracle rows are **diagnostics, not systems**: an oracle VAD is future
information and an oracle count is unavailable at inference. They are here to
separate failure modes.

Three things worth stating explicitly:

**Miss and false alarm are identical across every non-oracle row.** All systems
share one causal VAD and one window grid, so the entire between-method difference
is *speaker confusion*. That is a useful sanity check on the harness and it means
the comparison is clean.

**"Offline beats online" is false as stated.** Offline AHC is 0.0283 *worse* than
online; offline spectral is 0.0283 *better*. The spread between the two standard
offline algorithms (0.0565) is exactly 2.0× the gap from online to the better of
them. **Which clustering algorithm you choose matters more than whether you run
online.** That is the most useful practical finding in this document, and it was
not the finding being looked for.

**The oracles barely help.** Oracle speaker count moves online DER by 0.0017
(ratio 0.16) and oracle VAD by 0.0070 (ratio 0.10). Both inside noise. Neither
"counting speakers" nor "detecting speech" is the bottleneck; speaker confusion in
the embedding space is.

### Statistical verdicts

`results/tables/statistical_tests.csv`. Paired Wilcoxon signed-rank + paired
bootstrap CI (2000 resamples) over the 24 recordings of **seed 0**, Holm-corrected
across the five-metric family `{der, confusion, jer, turn_accuracy,
speaker_count_error}`, then graded against `√2·σ_seed` from the 3-seed study.

DER:

| vs online | paired Δ | 95% CI | p_adj | ratio to noise | verdict |
|---|---|---|---|---|---|
| naive online | +0.0401 | [+0.005, +0.082] | 0.917 | 0.98 | inside noise |
| offline AHC | −0.0345 | [−0.061, −0.009] | 0.085 | 0.81 | inside noise |
| offline spectral | +0.0583 | [+0.013, +0.108] | 0.404 | 2.06 | not significant |
| online, oracle count | +0.0057 | [−0.013, +0.027] | 1.000 | 0.16 | inside noise |
| offline AHC, oracle count | −0.0298 | [−0.058, −0.000] | 0.138 | 0.63 | inside noise |
| online, oracle VAD | +0.0040 | [−0.003, +0.009] | 0.011 | 0.10 | inside noise |
| offline AHC, oracle VAD | −0.0097 | [−0.045, +0.032] | 0.320 | 0.24 | inside noise |

**Not one DER comparison survives.** Note the `online, oracle VAD` row: p_adj =
0.011, comfortably "significant", for a difference of 0.0040 that is 0.10× the
run-to-run noise scale. That is exactly the failure mode the noise gate exists to
catch — a paired per-recording test conditioning on one trained model per method
can be arbitrarily significant about a difference a different seed would erase.

Speaker count error, the one family member that does survive:

| vs online | online bias | other bias | paired Δ | p_adj | ratio | verdict |
|---|---|---|---|---|---|---|
| offline AHC | +0.67 | −0.63 | 1.292 | 0.0005 | 10.53 | **survives** |
| offline spectral | +0.67 | −0.58 | 1.250 | 0.0097 | 3.19 | **survives** |
| online, oracle count | +0.67 | −0.29 | 0.958 | 0.0107 | 5.32 | **survives** |
| offline AHC, oracle count | +0.67 | −0.92 | 1.583 | 0.0006 | 5.45 | **survives** |
| offline AHC, oracle VAD | +0.67 | −0.58 | 1.250 | 0.0007 | 13.89 | **survives** |
| naive online | +0.67 | +0.21 | 0.458 | 0.094 | 1.20 | not significant |

**The online tracker over-clusters and every offline method under-clusters.** This
is a real, large, reproducible difference — and it is a difference in *behaviour*,
not in accuracy, because it does not translate into a DER difference that survives.
A threshold-based online tracker that meets an ambiguous window creates a speaker;
a global clustering that meets the same ambiguity merges. Opposite inductive
biases, similar DER.

---

## 4. Ablations

`results/tables/ablation_summary.csv`, `ablation_tests.csv` ·
`results/figures/ablations.png`. One config switch each.

| ablation | DER (3 seeds) | `√2·σ_seed` | confusion | speakers | windows/decision | cost of removal (paired, seed 0) | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| full | 0.3020 | 0.027 | 0.1972 | 4.33 | 2.88 | — | — | — |
| `bounded_window: false` | 0.3160 | 0.056 | 0.2111 | 4.06 | **1.00** | +0.0025 | 0.04 | **inside noise** |
| `spawn_enabled: false` | 0.5245 | 0.020 | 0.4196 | 1.50 | 2.88 | +0.2314 | 8.70 | **survives** |
| `centroid_momentum: 1.0` | 0.4366 | 0.026 | 0.3317 | 4.82 | 2.88 | +0.1104 | 4.15 | **survives** |
| `causal: false` | 0.2880 | 0.027 | 0.1831 | 4.47 | 2.88 | −0.0228 | 0.86 | inside noise |

**`bounded_window: false` genuinely disables the mechanism** — the
windows-per-decision diagnostic drops from 2.88 to exactly 1.00, so the ablation
is not a no-op that failed to apply. The mechanism is off, and DER does not move.

**Spawning is essential.** Without it the tracker finds 1.50 speakers instead of
3.74 and DER nearly doubles. This is the largest effect in the repository and the
least surprising: a speaker who first talks after the warm-up prefix simply cannot
be represented.

**Centroid memory is essential.** `centroid_momentum: 1.0` sets each centroid to
the last embedding assigned to it, and DER rises by 0.110. An EMA centroid is a
low-variance estimate of the speaker; a single embedding is not.

**Causality is nearly free.** Centred padding gives each frame ~150 ms of future
context and buys 0.0228 DER at 0.86× the noise scale. Note that `noncausal` is
**not a system**: its reported emission latency understates its true latency by the
lookahead its padding uses, and `tests/test_causality.py` requires it to *fail* the
bit-identity check the shipped model passes. It is here to price the guarantee, and
the price is at most about one noise unit.

---

## 5. Calibration and abstention

`results/tables/calibration.csv`, `calibration_summary.csv` ·
`results/figures/calibration.png`. 4,774 turn decisions per method per seed;
659 excluded per seed (non-speech or overlapped regions, where a single-label
system has no correct answer). Temperature fitted on **dev** only.

| system | turn acc | mean conf | over/under | ECE | ECE after T | ACE | MCE | Brier | NLL | error AUROC | AURC | oracle AURC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| online | 0.7964 | 0.6252 | **−0.171** | 0.1779 | **0.0835** | 0.1723 | 0.6436 | 0.1801 | 0.6167 | 0.6754 | 0.1485 | 0.0226 |
| naive online | 0.8052 | 0.6389 | −0.166 | 0.1718 | 0.0943 | 0.1671 | 0.5589 | 0.1661 | 0.5642 | **0.7392** | 0.1105 | 0.0206 |
| offline AHC | 0.7584 | 0.8494 | **+0.091** | **0.0980** | 0.1432 | 0.0972 | 0.2117 | 0.1775 | 1.1026 | 0.6348 | 0.2179 | 0.0325 |

Abstention — error rate over the retained decisions:

| coverage | online | naive online | offline AHC |
|---|---|---|---|
| 100% | 0.2036 | 0.1948 | 0.2416 |
| 80% | 0.1536 | 0.1303 | 0.1807 |
| 50% | 0.1281 | 0.0915 | 0.1761 |

**What works.** Deferring the least-confident 20% of turns cuts the online error
rate from 0.204 to 0.154, a 25% relative reduction. That is the practical payoff of
the online design: a bounded-latency system that flags its own doubtful turns can
route them to the offline pass it is otherwise worse than.

**What does not, stated plainly:**

- **The naive baseline's confidence is a better error detector than the
  proposal's** — AUROC 0.739 vs 0.675, and AURC 0.111 vs 0.149. The micro-cluster
  query improves neither accuracy nor confidence quality.
- **Both online systems are materially under-confident** (−0.17), the offline one
  over-confident (+0.09). Opposite failure modes; a single recipe will not fix both.
- **Temperature scaling helps online and hurts offline.** Online ECE 0.178 → 0.084;
  offline AHC 0.098 → 0.143. The offline confidence is a *post hoc* similarity to
  the final cluster centroid — it was never available at decision time — and it is
  already near-calibrated, so a temperature fitted on dev NLL sharpens it past the
  optimum. The asymmetry is a property of the two confidence definitions, not of the
  clustering.
- **The gap to the oracle is large.** AURC 0.149 against an oracle floor of 0.023.
  A confidence that ranked errors perfectly would recover far more than this one
  does.

---

## 6. Efficiency

`results/tables/efficiency.csv`, `scaling.csv` · `results/figures/efficiency.png`.
Warm-up 8 iterations, 25 repeats, median and IQR — the standard's floor, because
too little warm-up once made a small model look 5× slower than reality.

| system | median emission delay | p95 | max | wall-clock / 60 s audio | IQR | RTF | clustering ms |
|---|---|---|---|---|---|---|---|
| online (B = 500 ms) | **500 ms** | 500 ms | 500 ms | 103.0 ms | 10.4 | 0.0017 | 42.5 |
| naive online | **0 ms** | 0 ms | 0 ms | 88.6 ms | 24.2 | 0.0015 | 25.4 |
| offline AHC | 29,500 ms | 56,200 ms | 59,000 ms | 82.2 ms | 12.3 | 0.0014 | 31.1 |
| offline spectral | 29,500 ms | 56,200 ms | 59,000 ms | 88.7 ms | 11.5 | 0.0015 | 25.1 |

**The efficiency claim, stated exactly: 59.0× lower median emission delay** (500 ms
vs 29,500 ms). This is an algorithmic property — a consequence of the dependency
structure, not of this laptop — and it grows without bound with recording length,
because the offline delay *is* the recording length while the online delay is the
budget.

**Compute is not the win at 60 s, and that is reported rather than hidden.** Online
total wall-clock is 1.25× *higher* than offline AHC: the tracker takes a decision
per window while AHC runs one batched matrix job, and at 225 windows the batched job
still wins. Both are far below real time (RTF ≈ 0.002), which is what makes the
emission-delay comparison a statement about algorithm design.

**Compute becomes the win at length.** `scaling.csv`, clustering step only, on
pre-computed embeddings:

| speech windows | audio | online | offline AHC | offline spectral |
|---|---|---|---|---|
| 58 | 15 s | 11.1 ms | 4.3 ms | 5.5 ms |
| 116 | 30 s | 30.9 ms | 10.0 ms | 10.3 ms |
| 231 | 60 s | 46.5 ms | 19.8 ms | 25.8 ms |
| 456 | 120 s | 77.1 ms | 113.2 ms | 104.1 ms |
| 912 | 240 s | **141.8 ms** | 728.9 ms | 525.8 ms |

Per window: online **0.191 → 0.156 ms** (flat, as designed), offline AHC
**0.074 → 0.799 ms** (a 10.9× rise over the same range). The crossover is at about
120 s of audio; at 240 s online clusters **5.1× faster** and the ratio keeps
growing. Resident state is `max_speakers × embed_dim` floats and never grows with
length.

---

## Retractions

This section is not a formality. The claim the project was designed around is here.

### Retracted: the DER-versus-latency tradeoff curve

**The claim.** That a bounded-latency online diarizer pays a quantifiable accuracy
cost that decreases with the latency budget, producing a measurable `DER(B)` curve
that approaches the offline asymptote.

**What was measured.** Across nine budgets from 0 to 16,000 ms, over three seeds
and 24 recordings each, the DER range is **0.0178** against a run-to-run noise
scale of **0.0558**. Ratio **0.32**. The curve is flat inside noise.
`results/tables/latency_curve.csv`.

**Status: retracted.** The curve is shipped, plotted with its noise band, and
described as flat. The shape it appears to have (minimum near 500–1000 ms, rising
beyond 2000 ms) is stated as unconfirmed, and the supporting evidence for a real
mechanism is the monotone speaker-count inflation with budget — which is legible
even though the DER consequence is not.

### Retracted: the bounded-lookahead micro-cluster mechanism

**The claim.** That using the within-budget lookahead to form a local
micro-cluster, and querying the speaker centroids with its mean instead of a single
window embedding, reduces DER by reducing query variance.

**What was measured.** `bounded_window: false` — which provably disables the
mechanism, since windows-per-decision drops from 2.88 to exactly 1.00 — changes DER
by **0.0025 (paired, seed 0)**, a ratio of **0.04** to the noise scale. On 3-seed
means the gap is 0.0140, still 0.25×.

**Status: retracted.** The mechanism is correctly implemented (its geometric
variance-reduction property is asserted directly in
`tests/test_online.py::test_averaging_reduces_noise`), it is verifiably engaged,
and it does not measurably help. The variance reduction is real; it just does not
convert into label accuracy once the centroid is itself a low-variance EMA
estimate.

### Disclosed: the momentum was retuned after seeing a test result

This is a form of test-set feedback and it is disclosed rather than hidden.

The first experiment pass tuned the three clustering thresholds on dev but left
`centroid_momentum` at a guessed 0.35. That pass produced a **dramatic** apparent
tradeoff — online DER 0.3784 at `B = 0` falling to 0.2977 at `B = 500 ms` — and
also showed the naive baseline (which uses an exact running mean) beating the
proposal at its own optimum. The second observation is a symptom of an untuned
parameter, so the momentum was swept on **dev, seed 0** across the whole budget
grid (`results/tables/momentum_tuning.csv`):

| momentum | B=0 | B=250 | B=500 | B=1000 | B=2000 |
|---|---|---|---|---|---|
| 0.05 | 0.3356 | 0.3004 | 0.2855 | 0.2898 | 0.2863 |
| **0.10** | **0.3239** | 0.3208 | 0.2775 | 0.2724 | **0.2590** |
| 0.20 | 0.3429 | 0.3020 | 0.2678 | 0.2920 | 0.2854 |
| 0.35 *(guessed)* | 0.3890 | 0.3452 | 0.3075 | 0.2954 | 0.3109 |
| 0.60 | 0.4774 | 0.4373 | 0.3498 | 0.3572 | 0.3994 |
| 1.00 | 0.5332 | 0.4471 | 0.4107 | 0.5037 | 0.5258 |

0.10 was selected as the best *average* over the whole budget grid, so the swept
axis is not itself tuned. With it, the tradeoff curve went flat.

**The honest reading: the dramatic first-pass curve was the lookahead compensating
for a badly-chosen centroid update rule, not latency buying accuracy.** That
statement is backed by re-running the whole sweep on **test** at momentum 0.35,
committed as `results/tables/latency_sweep_momentum035.csv` and summarised in
`latency_curve_momentum035.csv`:

| budget (ms) | DER @ momentum **0.35** (guessed) | DER @ momentum **0.10** (tuned) | difference |
|---|---|---|---|
| 0 | 0.3729 | 0.3160 | **−0.0569** |
| 250 | 0.3163 | 0.3059 | −0.0104 |
| 500 | 0.3054 | 0.3020 | −0.0034 |
| 1000 | **0.3008** | **0.3013** | +0.0005 |
| 2000 | 0.3245 | 0.3049 | −0.0196 |
| 4000 | 0.3505 | 0.3191 | −0.0314 |
| 8000 | 0.3607 | 0.3159 | −0.0448 |
| range across budgets | **0.0721** | **0.0178** | |
| largest noise scale | 0.0699 | 0.0558 | |
| **range / noise** | **1.03** | **0.32** | |

Two things fall out of that table, and together they are the whole argument:

1. **The apparent size of the tradeoff is a function of how badly the rest of the
   tracker is tuned.** At momentum 0.35 the curve spans 1.03× the noise scale; at
   0.10 it spans 0.32×. Neither clears the 2× bar this repository requires for
   `survives`, but the direction is unambiguous.
2. **The two configurations are indistinguishable at the optimum** (0.3008 vs
   0.3013 at B = 1000 ms) and differ by 0.0569 at B = 0. The tuned centroid does not
   make the system better at its best budget — it removes the *need* for a budget.
   That is exactly what "the lookahead was compensating" means, quantified.

The tuning itself used only dev data. What was informed by a test observation is
the *decision to tune*, and this paragraph exists so a reader can discount
accordingly.

### Not supported: "the offline reference is more accurate"

Offline AHC scores 0.3303 against the online 0.3020 — the online system wins. Only
offline spectral (0.2738) is better, and that comparison is `not significant`
(p_adj 0.404, ratio 2.06). The framing this project started from — offline will beat
you on DER, measure the gap — turned out to be too coarse: **the choice of offline
clustering algorithm (spread 0.0565) matters twice as much as online-versus-offline
(gap 0.0283).**

### Not supported: any DER-level difference between methods

Zero of seven DER comparisons survive the noise gate. Two are nominally significant
after Holm correction (`online_oracle_vad` at p_adj 0.011 for a 0.10× noise-scale
difference; `offline_ahc` on JER at p_adj 0.041) and both are rejected by the noise
gate. If this document reported p-values alone it would claim results it cannot
support.

### Negative result: the proposal's confidence is worse than its baseline's

Error-detection AUROC 0.6754 for online against **0.7392** for naive online, and
AURC 0.1485 against 0.1105. The bounded-window query neither improves accuracy nor
improves the system's knowledge of its own errors.

### Negative result: temperature scaling degrades the offline system

Offline AHC ECE rises from 0.0980 to 0.1432 after a dev-fitted temperature, while
online falls from 0.1779 to 0.0835. The offline post-hoc confidence is already
near-calibrated and the NLL fit over-sharpens it.

---

## What would change these conclusions

Stated as predictions, none of them run here:

1. **More seeds.** The binding constraint on every retraction above is a noise
   scale of 0.03–0.06 DER from three seeds. Ten seeds would shrink it by ~1.8× and
   would be enough to resolve the 0.018 curve range.
2. **A stronger embedder.** The mechanism's benefit is variance reduction on the
   query. A lower-variance embedder should need *less* lookahead, so the curve's
   optimum should move left and the mechanism should matter even less — the opposite
   of what a "scale it up and it will work" reading would predict. `notebooks/05`
   sets up that run.
3. **Longer recordings.** All the compute advantages grow with length and all the
   accuracy comparisons were made at 60 s, where offline clustering is cheap. At
   240 s the compute crossover has already happened; the accuracy comparison at that
   length is unmeasured.
4. **Multi-label output.** The 0.0749 overlap floor is 25% of the online system's
   total DER. Overlap-aware assignment is the largest single available accuracy
   gain, and it is orthogonal to the latency question.
