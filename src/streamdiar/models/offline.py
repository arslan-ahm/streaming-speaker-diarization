"""The offline reference approach: global clustering over all embeddings at once.

This is the thing the project argues against, implemented properly so the
argument is falsifiable. Both standard variants are here — agglomerative
hierarchical clustering (AHC) and spectral clustering with the eigengap
heuristic — because they are the two things practitioners actually run, and
because they fail differently: AHC's threshold controls the speaker count
implicitly and drifts with recording difficulty, while spectral estimates the
count explicitly from the Laplacian spectrum.

**These baselines are deliberately made strong.** They consume the *same*
embeddings from the *same* trained model, use the *same* causal VAD, and the
spectral variant includes the row-thresholding affinity refinement of Wang et al.
(2018), which is a real improvement over a plain cosine affinity. A weak baseline
would make the latency curve look better and would be worthless. The expected —
and observed — outcome is that offline wins on DER; the deliverable is the size
of that gap as a function of latency, with these systems as the horizontal
reference line at infinite latency.

Latency, stated exactly
-----------------------
Every window's ``emission_frame`` is the recording's **final** frame, because
global clustering cannot assign any label before it has seen the last embedding.
So the emission delay of window ``k`` is ``n_frames - ends[k]``, which averages
half the recording length and grows without bound. That is not a modelling
choice; it is what the algorithm's dependency structure forces, and it is the
number the DER-versus-latency figure plots the offline line at.

Everything is hand-rolled on NumPy — no SciPy (this environment's copy fails to
import), no sklearn. Average-linkage AHC is the Lance-Williams recurrence,
spectral clustering is a symmetric eigendecomposition plus k-means++ with
deterministic seeded restarts. At the scale here (~240 windows per recording) the
O(n^3) merge loop is vectorised per merge and costs single-digit milliseconds.

References
----------
Wang et al., *Speaker Diarization with LSTM*, ICASSP 2018 — the spectral
refinement sequence and eigengap speaker counting.
Sell & Garcia-Romero, *Speaker Diarization with PLDA i-Vector Scoring and
Unsupervised Calibration*, SLT 2014 — AHC as the standard clustering baseline.
Ng, Jordan & Weiss, *On Spectral Clustering*, NeurIPS 2001 — normalised Laplacian.
Arthur & Vassilvitskii, *k-means++*, SODA 2007.
"""

from __future__ import annotations

import numpy as np

from ..config import DiarizerConfig
from .online import DiarizationOutput


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """``(n, n)`` cosine distance ``1 - sim``, clipped to ``[0, 2]``.

    The inputs are already L2-normalised by the embedder, so this is one matmul.
    The clip guards the diagonal against ``-1e-16`` style values that would make
    a distance negative and let AHC merge a point with itself first.
    """
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"expected (n, d) embeddings, got {e.shape}")
    return np.clip(1.0 - e @ e.T, 0.0, 2.0)


def agglomerative_average_linkage(
    distances: np.ndarray, threshold: float, max_clusters: int | None = None
) -> np.ndarray:
    """Average-linkage AHC, stopping when the closest pair exceeds ``threshold``.

    Args:
        distances: ``(n, n)`` symmetric distance matrix.
        threshold: Merge while the minimum inter-cluster distance is ``<=`` this.
        max_clusters: If given, keep merging past the threshold until at most this
            many clusters remain. Used by the oracle-count offline variant.

    Returns:
        ``(n,)`` cluster labels, densified to ``0..K-1`` in order of first
        appearance.

    The Lance-Williams recurrence for average linkage (UPGMA) is
    ``d(i u j, k) = (n_i d(i,k) + n_j d(j,k)) / (n_i + n_j)``, which is exact —
    the merged cluster's average distance is recovered without revisiting the
    members, which is what keeps each merge O(n) instead of O(n^2).
    """
    d = np.array(distances, dtype=np.float64, copy=True)
    n = d.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    np.fill_diagonal(d, np.inf)
    sizes = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    labels = np.arange(n, dtype=np.int64)

    while active.sum() > 1:
        flat = int(np.argmin(d))
        i, j = divmod(flat, n)
        best = d[i, j]
        if not np.isfinite(best):
            break
        n_clusters = int(active.sum())
        over_threshold = best > threshold
        if over_threshold and (max_clusters is None or n_clusters <= int(max_clusters)):
            break
        if i > j:
            i, j = j, i

        merged = (sizes[i] * d[i, :] + sizes[j] * d[j, :]) / (sizes[i] + sizes[j])
        d[i, :] = merged
        d[:, i] = merged
        d[i, i] = np.inf
        d[j, :] = np.inf
        d[:, j] = np.inf
        sizes[i] += sizes[j]
        active[j] = False
        labels[labels == j] = i

    return _densify(labels)


def _densify(labels: np.ndarray) -> np.ndarray:
    """Relabel to ``0..K-1`` in order of first appearance."""
    order = {v: k for k, v in enumerate(dict.fromkeys(np.asarray(labels).tolist()))}
    return np.asarray([order[int(v)] for v in np.asarray(labels).tolist()], dtype=np.int64)


def kmeans(
    x: np.ndarray, k: int, seed: int = 0, n_restarts: int = 5, n_iters: int = 40
) -> np.ndarray:
    """k-means with k-means++ initialisation, best-of-``n_restarts`` by inertia.

    Deterministic given ``seed``: the restarts draw from a seeded generator, so
    two runs of the offline baseline produce bit-identical labels. That is what
    lets ``docs/REPRODUCIBILITY.md`` claim max-abs-difference 0.0 on the offline
    per-recording scores as well as on the online ones.

    An empty cluster is re-seeded at the point furthest from its own centre
    rather than left empty, which otherwise silently reduces the effective ``k``
    and makes the eigengap's speaker count a lie.
    """
    data = np.asarray(x, dtype=np.float64)
    n = data.shape[0]
    k = max(1, min(int(k), n))
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    best_labels = np.zeros(n, dtype=np.int64)
    best_inertia = np.inf
    for restart in range(max(1, int(n_restarts))):
        rng = np.random.default_rng([int(seed), int(restart)])
        centres = _kmeanspp_init(data, k, rng)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(int(n_iters)):
            d2 = ((data[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
            new_labels = np.argmin(d2, axis=1).astype(np.int64)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                sel = labels == c
                if sel.any():
                    centres[c] = data[sel].mean(axis=0)
                else:
                    far = int(np.argmax(d2.min(axis=1)))
                    centres[c] = data[far]
        d2 = ((data[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
        inertia = float(d2.min(axis=1).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def _kmeanspp_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: each next centre drawn proportional to squared distance."""
    n = data.shape[0]
    centres = np.empty((k, data.shape[1]), dtype=np.float64)
    centres[0] = data[int(rng.integers(n))]
    closest = ((data - centres[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centres[c] = data[int(rng.integers(n))]
        else:
            centres[c] = data[int(rng.choice(n, p=closest / total))]
        closest = np.minimum(closest, ((data - centres[c]) ** 2).sum(axis=1))
    return centres


def refine_affinity(embeddings: np.ndarray, percentile: float) -> np.ndarray:
    """Cosine affinity with row-wise thresholding and symmetrisation.

    The refinement of Wang et al. (2018): in each row keep only the largest
    ``1 - percentile`` fraction of similarities, zero the rest, then symmetrise
    with ``max(A, A.T)``. The effect is to delete the weak cross-speaker edges
    that a dense cosine affinity is full of, which is what makes the eigengap
    legible at all — on the raw affinity the spectrum has no clear gap and the
    speaker count comes out at ``max_k`` almost every time.

    ``percentile <= 0`` returns the plain non-negative affinity.
    """
    e = np.asarray(embeddings, dtype=np.float64)
    a = np.clip(e @ e.T, 0.0, 1.0)
    np.fill_diagonal(a, 0.0)
    if percentile <= 0.0 or a.shape[0] < 3:
        return a
    keep = max(1, int(round((1.0 - float(percentile)) * a.shape[0])))
    # Zero everything below each row's own keep-th largest value.
    cut = np.partition(a, -keep, axis=1)[:, -keep][:, None]
    a = np.where(a >= cut, a, 0.0)
    return np.maximum(a, a.T)


def estimate_n_speakers_eigengap(
    affinity: np.ndarray, max_k: int, min_k: int = 1
) -> tuple[int, np.ndarray]:
    """Speaker count from the largest gap in the normalised Laplacian spectrum.

    Returns ``(k, eigenvalues_ascending)``. The count is
    ``argmax_k (lambda_{k+1} - lambda_k) + 1`` over ``k`` in ``[min_k, max_k]``,
    the standard eigengap heuristic: a graph with ``k`` well-separated components
    has ``k`` near-zero eigenvalues and a jump after them.

    The heuristic is genuinely fragile and this is stated in ``docs/RESULTS.md``
    rather than discovered by a reader: it is what makes the offline spectral
    baseline's speaker-count accuracy *worse* than the online method's on this
    data, even though its DER is better.
    """
    a = np.asarray(affinity, dtype=np.float64)
    n = a.shape[0]
    if n <= 1:
        return max(1, n), np.zeros(n, dtype=np.float64)
    deg = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    lap = np.eye(n) - (a * inv_sqrt[:, None]) * inv_sqrt[None, :]
    # Symmetrise against accumulated float asymmetry so eigh stays valid.
    lap = 0.5 * (lap + lap.T)
    vals = np.linalg.eigvalsh(lap)
    hi = int(min(max(1, max_k), n - 1))
    lo = int(max(1, min_k))
    if hi <= lo:
        return lo, vals
    gaps = vals[lo:hi + 1] - vals[lo - 1:hi]
    return int(lo + int(np.argmax(gaps))), vals


def spectral_cluster(
    embeddings: np.ndarray,
    max_k: int,
    percentile: float,
    seed: int = 0,
    n_restarts: int = 5,
    n_speakers: int | None = None,
) -> tuple[np.ndarray, int]:
    """Normalised spectral clustering. Returns ``(labels, k_used)``.

    Rows of the top-``k`` eigenvector matrix are L2-normalised before k-means (the
    Ng-Jordan-Weiss normalisation), which puts every point on the unit sphere so
    k-means measures angle rather than the eigenvector magnitude that reflects
    only node degree.
    """
    e = np.asarray(embeddings, dtype=np.float64)
    n = e.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    if n == 1:
        return np.zeros(1, dtype=np.int64), 1

    a = refine_affinity(e, percentile)
    k, _ = estimate_n_speakers_eigengap(a, max_k)
    if n_speakers is not None:
        k = int(max(1, min(int(n_speakers), n)))

    deg = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    lap = np.eye(n) - (a * inv_sqrt[:, None]) * inv_sqrt[None, :]
    lap = 0.5 * (lap + lap.T)
    _, vecs = np.linalg.eigh(lap)
    emb = vecs[:, :k]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    labels = kmeans(emb, k, seed=seed, n_restarts=n_restarts)
    return _densify(labels), k


def diarize_offline(
    embeddings: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    is_speech: np.ndarray,
    cfg: DiarizerConfig,
    n_frames: int,
    hop_frames: int,
    method: str = "offline_ahc",
    seed: int = 0,
    oracle_n_speakers: int | None = None,
) -> DiarizationOutput:
    """Global clustering over every speech window at once — the reference approach.

    Args:
        method: ``"offline_ahc"`` or ``"offline_spectral"``.
        oracle_n_speakers: If given, the clustering is forced to this many
            speakers (AHC merges past its threshold; spectral overrides the
            eigengap). Used by the oracle-count offline variant.

    Confidence for the offline systems is the cosine similarity between a window
    and its own final cluster centroid, squashed through the same softmax-style
    scale the online method uses so the calibration comparison is not a
    comparison of two different confidence definitions. It is a *post hoc* score
    and it is labelled as such: unlike the online margin it was not available at
    decision time, because there was no decision time.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    n_windows = emb.shape[0]
    speech = np.asarray(is_speech, dtype=bool)
    labels = np.full(n_windows, -1, dtype=np.int64)
    confidence = np.full(n_windows, np.nan, dtype=np.float64)
    margin = np.full(n_windows, np.nan, dtype=np.float64)
    ends = np.asarray(window_ends, dtype=np.int64)
    # The defining property: nothing is emitted until the recording ends.
    final_frame = int(ends[-1]) if n_windows else 0
    emission = np.full(n_windows, final_frame, dtype=np.int64)

    idx = np.flatnonzero(speech)
    k_used = 0
    if idx.size:
        sub = emb[idx]
        if method == "offline_spectral":
            sub_labels, k_used = spectral_cluster(
                sub, cfg.offline_spectral_max_k, cfg.offline_spectral_percentile,
                seed=seed, n_restarts=cfg.offline_kmeans_restarts,
                n_speakers=oracle_n_speakers,
            )
        elif method == "offline_ahc":
            dist = cosine_distance_matrix(sub)
            sub_labels = agglomerative_average_linkage(
                dist, cfg.offline_ahc_threshold, max_clusters=oracle_n_speakers
            )
            k_used = int(sub_labels.max()) + 1 if sub_labels.size else 0
        else:
            raise ValueError(f"unknown offline method {method!r}")
        labels[idx] = sub_labels

        # Post hoc confidence: similarity to the assigned cluster's centroid,
        # against the best rival centroid, on the same scale as the online margin.
        centroids = np.zeros((max(k_used, 1), sub.shape[1]), dtype=np.float64)
        for c in range(max(k_used, 1)):
            sel = sub_labels == c
            if sel.any():
                v = sub[sel].mean(axis=0)
                nrm = float(np.linalg.norm(v))
                centroids[c] = v / nrm if nrm > 1e-12 else v
        sims = sub @ centroids.T
        t = max(float(cfg.confidence_temperature), 1e-6)
        shifted = (sims - sims.max(axis=1, keepdims=True)) / t
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=1, keepdims=True)
        own = sims[np.arange(sub.shape[0]), sub_labels]
        rival = sims.copy()
        rival[np.arange(sub.shape[0]), sub_labels] = -np.inf
        best_rival = rival.max(axis=1) if rival.shape[1] > 1 else np.full(sub.shape[0], -np.inf)
        confidence[idx] = probs[np.arange(sub.shape[0]), sub_labels]
        margin[idx] = np.where(np.isfinite(best_rival), own - best_rival, own)

    used = {int(v) for v in labels.tolist() if v >= 0}
    return DiarizationOutput(
        labels=labels,
        window_starts=np.asarray(window_starts, dtype=np.int64),
        window_ends=ends,
        emission_frames=emission,
        confidence=confidence,
        margin=margin,
        spawned=np.zeros(n_windows, dtype=bool),
        query_size=np.ones(n_windows, dtype=np.int64),
        n_frames=int(n_frames),
        hop_frames=int(hop_frames),
        n_speakers=len(used),
        method=method,
        extra={"k_estimated": float(k_used)},
    )
