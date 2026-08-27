# Method

This document is the maths and the design reasoning. Measured numbers live in
[RESULTS.md](RESULTS.md); exact commands and timings live in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## 1. The problem, and why the reference approach cannot solve it

Speaker diarization answers "who spoke when". The dominant approach — used by
essentially every strong system since Sell & Garcia-Romero (SLT 2014) — is:

1. slide a window over the recording and extract a speaker embedding per window;
2. build the full `N x N` similarity matrix over those embeddings;
3. cluster it **globally**, with agglomerative clustering or spectral clustering
   (Wang et al., ICASSP 2018);
4. map cluster ids back onto time.

Step 3 is why it works and why it is useless for live captioning. Global
clustering sees every embedding before it assigns any label, so it can resolve an
ambiguous window at minute 2 using evidence from minute 40. It also cannot emit
*anything* until the recording ends: the first label's latency is the duration of
the recording.

This repository asks what happens when you take that away. Formally, let
`e_1..e_K` be the window embeddings, and let `L_k` be the label emitted for window
`k`. The offline system computes

    L_k = f(e_1, ..., e_K)              for every k

while a bounded-latency online system with budget `B` computes

    L_k = f(e_1, ..., e_{k + floor(B / hop)})

and must commit to `L_k` at time `end_k + B`, never revising it. The offline
system is the `B -> infinity` limit. The deliverable is not a win over it; it is
the **cost curve** `DER(B)`, with `DER(infinity)` as the asymptote.

### Why "no revision" is the right constraint

An alternative online formulation lets the system relabel the past as evidence
accumulates. That is a legitimate design (and is what a re-clustering streaming
system does), but it is a different product: a live caption already shown to a
user cannot be un-shown. Committing irrevocably at the deadline is the harder and
more honest constraint, and it is what makes the latency axis a single number
rather than a convergence curve per window.

---

## 2. Data: exact ground truth by construction

The shipped results come from a procedural generator
(`src/streamdiar/data/generator.py`). This is a deliberate methodological choice,
not a convenience.

**The reason is measurement resolution.** AMI and VoxConverse references are human
annotations whose boundary error is on the order of tens of milliseconds. The
differences this project needs to resolve — what a 250 ms change in latency budget
does to DER — are the same order. Scoring a boundary-sensitive metric against a
reference with boundary noise of comparable size measures the annotator, not the
system. Here the turn list is **sampled first** and the features are **rendered
from it**, so the reference is generative truth with zero annotation error.

A real-data path exists (`src/streamdiar/data/real.py`, WAV + RTTM, hand-rolled
log-mel front end) and is documented, but nothing in the tests or the shipped
tables depends on it.

### The generative model

A frame is a log-power spectrum-like vector in `R^F` (`F = 40`). Three additive
terms live in that space and **only the first carries identity**:

| term | form | role |
|---|---|---|
| timbre | `B_spk c_s`, `c_s in R^8` | the speaker. `B_spk` is a *global* smooth cosine basis, so identities lie on an 8-dimensional manifold |
| phonetic content | `B_phn d_p`, resampled every 8–20 frames | nuisance, **shared across speakers** |
| channel + energy | isotropic offset in `R^F` plus a slow scalar contour | nuisance, drawn per segment |

Three consequences worth stating:

**Identity is low-dimensional, nuisance is not.** `channel_sd = 1.5` is isotropic
in all 40 dimensions while timbre lives in an 8-dimensional subspace with
`speaker_scale = 0.5`. So the channel cannot be removed by normalisation, and the
only way to recover identity is to *learn* the projection onto the speaker
manifold. This is what makes training necessary rather than decorative — see §7.

**Overlap sums in the linear power domain.** Two simultaneous speakers combine as
`log(exp(a) + exp(b))`, which is what actually happens to a mel filterbank when two
people talk at once. Summing in the log domain instead would make overlap a
trivially detectable amplitude spike. Doubling the power adds only `log 2 ~ 0.69`,
and `tests/test_data.py::test_overlap_is_summed_in_the_power_domain` pins it.

**Turn structure is log-normal with no self-succession.** Turn durations are
log-normal (the standard family for conversational turns). Speaker choice is a
first-order chain that forbids a speaker following themselves, because two
consecutive turns by one speaker *are* one turn — forbidding it makes the sampled
log-normal the true marginal turn-length distribution.

Turns and acoustics draw from **separate RNG streams** addressed by name, so
changing the noise level leaves the turn structure bit-identical. That is what
makes a difficulty sweep a controlled experiment rather than a new dataset.

---

## 3. The causal embedder

`src/streamdiar/models/embedder.py`. A dilated convolution stack (Bai et al.,
2018) with statistics pooling to a fixed-size embedding (Snyder et al., ICASSP
2018), 52,144 parameters, receptive field 31 frames = 310 ms.

### Causality is structural, not a convention

Three things in a normal speaker-embedding stack quietly read the future, and all
three are removed:

1. **Convolution padding.** A `Conv1d` with `padding=(k-1)d//2` is *centred*:
   output `t` depends on inputs up to `t + (k-1)d/2`. Here the input is
   left-padded by the full `(k-1)d` and the convolution is unpadded, so output `t`
   depends on inputs `<= t` exactly.
2. **Normalisation over time.** `BatchNorm1d` and `GroupNorm` on a `(B, C, T)`
   tensor both reduce over `T`, so one late frame shifts every earlier activation.
   The norm here is a `LayerNorm` over the **channel** axis at each timestep
   independently — no time reduction at all. This is the same class of constraint
   as "GroupNorm not BatchNorm because EMA averages parameters and not buffers":
   the normalisation choice is forced by the guarantee, not by accuracy.
3. **Input standardisation.** Normalising a recording by its own mean and variance
   is the most common causality leak in diarization code, and it is invisible
   because it *improves* offline numbers. Per-channel statistics here are
   estimated once from **training** segments and frozen into buffers that travel
   with the checkpoint.

The consequence is asserted directly rather than argued: replace every frame after
`t0` with noise, and the embeddings of all windows ending at or before `t0` are
**bit-identical** — max absolute difference exactly `0.0`, not "close".
`tests/test_causality.py` checks this at three levels (conv stack, VAD, diarizer)
and requires the `causal=False` variant to **fail** the same check, so the clean
result is not vacuous.

### O(1)-per-window pooling

Statistics pooling needs the mean and standard deviation of the frame features in
each window. Naively that is `O(window)` per window. Prefix sums of the features
and their squares make it `O(C)` per window regardless of window length:

    mean = (S[e] - S[s]) / n
    var  = (Q[e] - Q[s]) / n - mean^2

The prefix sums are accumulated in **float64**. This is not caution for its own
sake: in float32 at 6000 frames the cancellation in `E[x^2] - E[x]^2` produced
variances around `-1e-4` during development, so `sqrt` returned NaN and every
downstream embedding became NaN. Widening the accumulator removes the cause; the
clamp before `sqrt` covers the residual.

Prefix sums do not break causality: `S[e]` depends only on frames `< e`, and the
scan order is fixed by index, so the prefix values are bit-identical when later
frames change. `test_prefix_sum_pooling_does_not_leak_the_future` checks it by
appending 500 frames and comparing.

---

## 4. Metrics

### DER with an optimal one-to-one mapping

On a frame grid, with `R_t` active reference speakers, `H_t` active hypothesis
speakers, and `C_t` reference speakers whose *mapped* hypothesis label is also
active at `t` (NIST md-eval):

    miss_t      = max(0, R_t - H_t)
    falarm_t    = max(0, H_t - R_t)
    confusion_t = min(R_t, H_t) - C_t
    DER         = sum_t (miss_t + falarm_t + confusion_t) / sum_t R_t

**The denominator is reference speaker-time, counted with multiplicity.** A frame
with two simultaneous speakers contributes 2. This is why DER can exceed 1.0, and
why a system that emits one speaker per frame has an irreducible miss floor equal
to the overlap fraction.

**The mapping is global, optimal, and computed with a hand-rolled Hungarian
solver** (`metrics/assignment.py`, Jonker & Volgenant, *Computing* 38, 1987, with
dual potentials). Greedy label mapping is not merely suboptimal, it *inflates DER*,
and the counterexample is small enough to be embarrassing. With co-occurrence

            hyp0  hyp1
    ref A     10     9
    ref B      9     0

greedy takes `A -> 0` (10), leaving B with 0, total 10. The optimum is
`A -> 1, B -> 0`, total 18. Embedded in a real 28-frame reference this is the
difference between `DER = 10/28 = 0.357` and `DER = 18/28 = 0.643`.
`tests/test_der.py::TestOptimalMappingMatters` pins the whole example.

There is no SciPy dependency: the copy in this environment ships without
`scipy.__config__` and fails to import, which is exactly the install fragility to
avoid. So the solver is validated against **brute-force enumeration of every
permutation** on 400 random matrices up to 5x5 — a stronger check than agreeing
with another library, because enumeration *is* the definition of the optimum.
Where SciPy happens to be importable the tests additionally cross-check against
it, but they skip cleanly when it is not.

#### An aside on what the mapping optimises

md-eval chooses the mapping to maximise `sum_t C_t`, i.e. to minimise confusion
only. That is not obviously the same as minimising total DER — but it is, because
`miss_t` and `falarm_t` depend only on the *counts* `R_t` and `H_t`, not on which
label is matched to which. The two objectives therefore coincide, and no
generality is lost.

### JER

Jaccard Error Rate (Ryant et al., Interspeech 2019) averages a per-reference-
speaker error over **speakers** rather than over time, using the same mapping:
`JER_i = (miss_i + falarm_i) / total_i`, with an unmapped reference speaker scoring
1.0. It disagrees with DER on purpose. A recording where speaker A talks for 90 s
and speaker B for 10 s, hypothesised as one speaker, scores `DER = 0.10` and
`JER > 0.5`: JER refuses to let a dominant talker hide a completely missed one.

### Latency

Two quantities, kept separate because conflating them is how streaming claims get
fudged:

* **Emission delay** — audio time between a window's audio finishing arriving and
  its label being emitted. A property of the algorithm's *dependency structure*,
  so it is exact, machine-independent and reproducible. For the online system it
  is the budget; for the offline system it is `n_frames - end_k`, which grows
  without bound.
* **Real-time factor** — compute time over audio duration. Reported so a reader
  can confirm neither system is compute-bound, which is what makes the emission-
  delay comparison a statement about algorithm design rather than about this
  laptop.

Latency statistics are computed over **turn decisions only**. Non-speech windows
are not decisions about a speaker, and their zero delay would flatter every
system.

### Undefined is NaN

A recording with no reference speech has no DER. Returning 0.0 there would
silently pull down the mean of a sweep. Every aggregate reports
`n_contributing_<metric>` when any value was non-finite, because a DER averaged
over 22 of 24 recordings is a different number from one averaged over 24.

---

## 5. The online mechanism

`src/streamdiar/models/online.py`.

### State

A bounded set of at most `max_speakers` L2-normalised centroids, each with a
count. Resident state is `max_speakers x embed_dim` floats and **never grows with
recording length** — that is the bounded-memory half of the claim, and the cap is
hard: once it is reached no further speaker can be created and the assignment is
forced.

Centroids update by exponential moving average, `c <- normalize((1 - a) c + a e)`
with `a = centroid_momentum = 0.35`, not by running mean. The reason is
recoverability: a running mean makes a centroid progressively immovable, so a
speaker whose first few windows were mis-assigned can never be repaired. An EMA
keeps every centroid responsive at constant cost. The price is that the centroid
is not the mean of its members and therefore not a k-means fixed point — which
only matters if one wanted to claim optimality, and none is claimed. The
`no_momentum` ablation (`a = 1.0`, centroid = last embedding assigned) measures
whether the memory earns its place.

### What the budget buys

A budget of `B` ms permits `m = floor(B / hop_ms)` further hops of lookahead. When
window `k`'s deadline arrives, the pending buffer holds the speech windows in
`[k, k + m]`, and the decision for the oldest one is made from a **local
micro-cluster** rather than from a single noisy embedding:

1. Start at the oldest pending window. Walk forward in time, accumulating while
   each next embedding matches the running micro-cluster mean at cosine
   similarity `>= micro_cluster_threshold`.
2. **Stop at the first mismatch.** A speaker turn is contiguous; if the speaker
   changes and later changes back, the second run is a different turn and must not
   be averaged into *this* decision, or a short interjection would drag the query
   toward the wrong speaker.
3. Use the micro-cluster mean as the query against the tracked centroids.

The justification is variance reduction: averaging `m` window embeddings of the
same speaker cuts the query's noise by roughly `sqrt(m)`, and the assignment
threshold converts reduced query noise into fewer label errors.
`test_averaging_reduces_noise` asserts the geometric claim directly.

Setting `bounded_window = False` keeps the buffer and the delay but queries with
the single embedding. That is the ablation which isolates *the mechanism* from
*the mere fact of having a buffer* — an important distinction, because a variant
that also removed the delay would confound the two.

### Assignment and spawning

Given the query `q` and centroids `c_1..c_A`:

* if `A = 0`, or `max_i cos(q, c_i) < spawn_threshold` and spawning is permitted,
  create a new speaker seeded at the *oldest window's own* embedding;
* otherwise assign to `argmax_i cos(q, c_i)` and update that centroid.

The centroid updates with the individual embedding of the window being labelled,
not with the micro-cluster mean, so the other buffered windows are not
double-counted when they are emitted later. The lookahead denoises the **query**,
not the state.

Two switches are kept separate because they fail differently:

* `spawn_enabled = False` — new speakers may only be created during a warm-up
  prefix of `warmup_windows` windows. A speaker who first talks late in the
  recording can then never be represented.
* `oracle_n_speakers` — the true count is supplied and caps spawning. This
  separates "clustering is hard" from "counting speakers is hard".

Conflating those two is the most common confound in the online-diarization
literature, and it is why both appear as independent rows.

### Emission timing is exact, not modelled

Window `k` is emitted when window `k + m` arrives, so its emission frame is
`end_{k+m}` and its delay is exactly `m * hop` — except in the recording's tail,
where the remaining windows flush at the final frame and their delay is *smaller*.
Latency is therefore measured by `metrics/latency.py`, which raises if any
emission precedes its own window's end.

One subtlety, and it was a real bug: the deadline check runs on **every** window,
speech or not. Checking only on speech windows let a pending decision drift past
its deadline across a silence and quietly buy unbudgeted lookahead, which made the
`B = 0` curve point impossible. `test_decisions_do_not_drift_past_the_deadline`
pins it.

### Why the curve is flat below one hop

`m = floor(B / hop)`, so at `hop = 250 ms` every budget in `[0, 250)` buys nothing:
the next window does not exist yet. The sweep includes both 0 ms and 125 ms
specifically to make that step visible rather than letting a reader assume the
curve is smooth. Pretending a 125 ms budget helps would require partial windows
the streaming grid does not produce.

---

## 6. Baselines, and how they are kept fair

Every baseline runs in this codebase, on the same recordings, from the same trained
embedder, through the same VAD and the same scoring code. A baseline that is
described but not run is not a baseline.

| system | mechanism | latency |
|---|---|---|
| `online` | the above | budget |
| `naive_online` | greedy nearest-centroid, running-mean centroids, no buffer | 0 |
| `offline_ahc` | average-linkage AHC over all embeddings, threshold-based `K` | recording length |
| `offline_spectral` | normalised spectral clustering + eigengap `K` | recording length |

`naive_online` is a **baseline rather than an ablation**: it differs in the centroid
update rule as well as in lookahead. Comparing against it answers "is the proposed
mechanism better than the obvious online thing"; the `bounded_window=False`
ablation answers the narrower "is the lookahead the reason". Both questions are
reported, and they do not have the same answer.

The offline systems are deliberately made **strong**. `offline_spectral` includes
the row-thresholding affinity refinement of Wang et al. (2018), which is a real
improvement over a plain cosine affinity — without it the Laplacian spectrum has
no legible gap and the speaker count comes out at `max_k` almost every time. A weak
baseline would make the latency curve look better and would be worthless.

Everything is hand-rolled on NumPy: average-linkage AHC via the exact
Lance-Williams recurrence `d(i u j, k) = (n_i d(i,k) + n_j d(j,k)) / (n_i + n_j)`,
and spectral clustering as a symmetric eigendecomposition of
`I - D^{-1/2} A D^{-1/2}` followed by row-normalised k-means++ with seeded
deterministic restarts (Ng, Jordan & Weiss, NeurIPS 2001; Arthur &
Vassilvitskii, SODA 2007).

### Voice activity detection

Papers routinely report DER "with oracle VAD". That isolates the clustering
question, and it is reported here as a secondary column for exactly that reason —
but it cannot be the headline, because **an oracle VAD is future information**:
knowing frame `t` is speech generally requires having heard frame `t+1`. A system
claiming bounded latency cannot use it.

So the primary numbers use a strictly causal energy VAD: frame energy against a
threshold, a hangover that only ever extends a decision *forward* in time, and a
minimum onset run that suppresses single-frame spikes at the cost of detecting
onsets a few frames late. That onset delay is real and is included in the reported
latency, because it delays the frame labels. The threshold is one scalar **fitted
on the dev split**, not derived from the generator's constants — deriving it from
`BACKGROUND_LOG_POWER` would be leaking knowledge of the data-generating process
into the model, and would not transfer to the real-data path at all.

---

## 7. Hyperparameters: what was tuned, on what, and the mistake that motivated it

Three clustering thresholds were tuned by grid search on the **dev split, seed 0
only** (`scripts/tune_thresholds.py`); the chosen values are then frozen for every
seed and every experiment, so seeds 1 and 2 are genuinely held out with respect to
hyperparameter choice. The grids are committed to
`results/tables/threshold_tuning_*.csv`.

**Why this module exists.** The first version of this project guessed
`spawn_threshold = 0.55` from cosine-similarity intuition. Measured on dev, the
similarity distributions are:

| pair | median | 5th pct | 95th pct |
|---|---|---|---|
| same speaker, different window | **0.968** | 0.604 | — |
| different speakers | **0.641** | — | 0.956 |

The embedding space is a narrow cone, not an isotropic ball, and the best single
same/different threshold sits near **0.85**. At 0.55 a new speaker was essentially
never created: the diarizer found 1.62 speakers where there were 3.50, and DER was
0.497. After tuning, dev DER was 0.308. That is a factor-of-1.6 error caused
entirely by a guessed constant, and it is recorded rather than quietly fixed.

**The selection rule is not pure argmin.** On the online grid the best three
settings span a DER range of 0.003 — far inside the spread over 8 dev recordings —
while their speaker-count bias ranges from +0.75 to +2.5 speakers. Taking the DER
argmin would freeze a badly over-clustering configuration on the strength of a
difference that is not there. So the rule, applied identically to all three grids,
is: **minimum dev DER, with near-ties (within 1% relative DER) broken on the
smallest absolute speaker-count bias.** Frozen values: `spawn_threshold = 0.83`,
`micro_cluster_threshold = 0.85`, `offline_ahc_threshold = 0.22` (cosine distance),
`offline_spectral_percentile = 0.80`.

### Data difficulty is also a calibrated choice

The generator's difficulty was set by measuring what an *untrained* embedder
achieves on the 12-way batch task. At the original `speaker_scale = 1.0`,
`channel_sd = 0.35` a randomly initialised network already scored **0.868**, so the
experiment would have measured random projections rather than learned embeddings.
At the shipped `speaker_scale = 0.5`, `channel_sd = 1.5`, `noise_sd = 0.9`: untrained
**0.281**, trained **0.708** after 300 steps, chance 0.083. That gap is what makes
the embedder load-bearing.

---

## 8. Training objective

Prototypical / GE2E loss (Wan et al., ICASSP 2018; Snell et al., NeurIPS 2017). A
batch is `N` speakers x `M` segments; for segment `(n, m)`,

    S[(n,m), k] = w * cos(e_nm, c_k) + b,    c_k = mean_{m'} e_km'

with the target `n`, and one detail that is not optional: when `k = n` the
prototype **excludes the query itself**,

    c_n^{-(nm)} = (1 / (M - 1)) * sum_{m' != m} e_nm'

Without the exclusion, `e_nm` appears on both sides of its own positive similarity,
the loss can be driven down by inflating one embedding, and training collapses to a
state where every embedding of a speaker matches its own batch-mate mean while
different speakers are not separated. This is the most common GE2E
reimplementation bug and it produces a training curve that looks **better** than
the correct one — so it is pinned to a closed form:
on orthogonal two-segment speakers the correct loss is
`log(1 + exp(-1/sqrt(2))) = 0.400834`, and the inclusive-prototype bug gives
`0.217706`.

`w` and `b` are learnable, initialised at 10 and -5. Cosine similarities live in
`[-1, 1]`, and a softmax over that range is nearly uniform, so without a learned
scale the gradient signal is tiny and the model underfits at any learning rate.

An angular-margin softmax (AAM/ArcFace) was not used: it needs a weight column per
training speaker, tying model size to the pool, and it gives nothing at this
scale. The prototypical form also matches the test-time operation — comparing an
embedding to a centroid — exactly.

Training segments get **independent channel offsets even within a speaker**,
specifically so channel cannot be a shortcut for identity. Overlapped windows are
not represented in training, because nothing sensible can be learned for them with
a single-label objective; the resulting overlap-region error is reported separately
rather than hidden.

---

## 9. Confidence, calibration, abstention

For each turn decision the competitor set is `{existing centroids} + {spawn}`,
where the virtual spawn option is scored at `spawn_threshold`. Confidence is the
softmax over that set at `confidence_temperature`; the margin
`chosen - best_alternative` is the logit that temperature scaling recalibrates.

Including the virtual spawn option matters: without it, every decision taken while
only one centroid exists would report confidence 1.0 by vacuity, and those are a
large fraction of a short recording's decisions. The genuinely *first* decision of
a recording still reports 1.0 — there is no alternative — and that handful of
windows is a documented artefact rather than a hidden one.

**Correctness** for calibration is defined through the DER mapping: a decision is
correct when its emitted label maps to the reference speaker active in that
window's region. Windows with no reference speaker (non-speech) or several
(overlap) are excluded and **counted** (`n_excluded`), because a single-label
system cannot be right in the sense being measured.

Temperature is fitted on **dev** by grid search on NLL, never on test. Reported:
ECE (equal-width, Guo et al., ICML 2017), ACE (equal-mass, Nixon et al., CVPRW
2019 — higher and more informative for a sharp model), MCE, Brier, NLL,
risk-coverage and AURC against the oracle floor (Geifman & El-Yaniv, NeurIPS
2017), and error-detection AUROC via the Mann-Whitney identity.

For the offline systems the analogous confidence is a *post hoc* similarity to the
assigned cluster centroid. It was not available at decision time because there was
no decision time, and that asymmetry is stated wherever those numbers appear.

---

## 10. Statistics

`src/streamdiar/metrics/stats.py`, hand-rolled (no SciPy).

Every difference is placed against **two** scales:

1. **Per-recording spread** — paired Wilcoxon signed-rank plus a paired bootstrap
   CI (>= 2000 resamples) over the test recordings, Holm-Bonferroni corrected
   across the five-metric family `{der, confusion, jer, turn_accuracy,
   speaker_count_error}`, with paired Cohen's d.
2. **Run-to-run spread** — the seed study. Two runs of the same config differ with
   standard deviation `sqrt(2) * sd` where `sd` is the across-seed sd of a single
   run. Comparing a between-method delta against a bare `sd` understates the noise
   by 29%, which is enough on its own to turn "inside noise" into "significant".

The `verdict()` function puts the noise scale **ahead** of the p-value:

* `inside noise` — `|delta| < sqrt(2) sd`, whatever the p-value. Checked first.
* `survives` — corrected `p < 0.05` **and** `|delta| >= 2 sqrt(2) sd`.
* `suggestive` — corrected `p < 0.05` but only 1–2x the noise scale.
* `not significant` — above the noise scale but the paired test does not reject.

**Unit of analysis, stated because it constrains every claim.** A paired
per-recording test conditions on *one trained model per method*. It answers "do
these two sets of weights differ on this test set", not "is this method better".
A method-level claim needs the training run as the sampling unit, and three seeds
is not enough power for that — which is precisely why the noise-scale column, not
the p-value, decides the verdict. The test recordings are also generated per seed,
so paired tests are run **within** a seed; pooling seeds would break the pairing.

The Wilcoxon implementation uses the **exact** null distribution (dynamic
programming over all `2^n` sign patterns) when `n <= 25` and the non-zero absolute
differences are distinct, and a normal approximation with the standard tie
correction `var = n(n+1)(2n+1)/24 - sum(t^3 - t)/48` and a 0.5 continuity
correction otherwise. Silently using the exact test when ties are present is a real
and common bug. It is validated against full enumeration on 25 random cases and
against R's published value for the textbook paired sample (`V = 40`,
`p = 0.03906`).
