"""Figures. Every panel reads a committed CSV — none recomputes a number.

That constraint is the point: if a figure disagrees with a table, the figure is
wrong, and the only way to guarantee they agree is to have one source. Each
function takes a DataFrame loaded from ``results/tables/`` and returns the path it
wrote.

The backend is chosen defensively rather than forced at import time. Calling
``matplotlib.use("Agg")`` unconditionally at module scope makes every notebook
plot render blank, which cost the sibling project a full set of empty figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if "ipykernel" not in sys.modules:  # pragma: no cover - environment dependent
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

#: One colour per system, used consistently across every figure so the reader
#: does not have to re-learn the legend.
COLORS = {
    "online": "#1f77b4",
    "naive_online": "#ff7f0e",
    "offline_ahc": "#2ca02c",
    "offline_spectral": "#9467bd",
    "online_oracle_count": "#17becf",
    "offline_ahc_oracle_count": "#8c564b",
    "online_oracle_vad": "#7f7f7f",
    "offline_ahc_oracle_vad": "#bcbd22",
}
FIG_DIR = Path("results/figures")


def _save(fig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def der_vs_latency(
    curve: pd.DataFrame, out: str | Path = FIG_DIR / "der_vs_latency.png"
) -> Path:
    """**The headline figure.** DER against the latency budget, offline as a reference line.

    Two panels. Left: DER versus budget on a symmetric-log x-axis so the ``B = 0``
    point is visible alongside 16 s. The shaded band is ``+/- sqrt(2) * sd`` across
    seeds — the run-to-run scale — so a reader can see immediately which parts of
    the curve are structure and which are wobble. The offline systems appear as
    horizontal dashed lines: they are the infinite-latency asymptote, not
    competitors.

    Right: the same curve against *measured* median emission delay rather than the
    configured budget, with the offline systems plotted at their own measured
    delay. This is the panel that shows what the design actually buys — the
    offline points sit three orders of magnitude to the right.
    """
    online = curve[curve["method"] == "online"].sort_values("latency_budget_ms")
    online = online[np.isfinite(online["latency_budget_ms"])]
    offline = curve[curve["method"].str.startswith("offline")]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    x = online["latency_budget_ms"].to_numpy(dtype=float)
    y = online["der"].to_numpy(dtype=float)
    band = online.get("der_noise_scale")
    ax.plot(x, y, "o-", color=COLORS["online"], lw=2, ms=6,
            label="online (bounded latency)")
    if band is not None:
        b = np.nan_to_num(band.to_numpy(dtype=float), nan=0.0)
        ax.fill_between(x, y - b, y + b, color=COLORS["online"], alpha=0.18,
                        label=r"$\pm\sqrt{2}\,\sigma_{\rm seed}$")
    for _, row in offline.iterrows():
        m = str(row["method"])
        ax.axhline(float(row["der"]), ls="--", lw=1.6, color=COLORS.get(m, "k"),
                   label=f"{m} (offline, infinite latency)")
    ax.set_xscale("symlog", linthresh=125)
    # symlog draws a mirrored negative decade by default; clip it, since a
    # negative latency budget is not a thing.
    ax.set_xlim(-10, float(x.max()) * 1.6 if x.size else 1.0)
    ax.set_xlabel("latency budget (ms, symlog)")
    ax.set_ylabel("DER (collar 0, overlap scored)")
    ax.set_title("DER versus latency budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")

    ax = axes[1]
    # A log axis cannot show 0, so the B=0 point is drawn at 1 ms and labelled as
    # such rather than silently relocated.
    measured = np.maximum(online["latency_median_ms"].to_numpy(dtype=float), 1.0)
    ax.plot(measured, y, "o-", color=COLORS["online"], lw=2, ms=6, label="online")
    if measured.size and float(online["latency_median_ms"].min()) <= 0.0:
        ax.annotate("B = 0 (drawn at 1 ms;\na log axis cannot show 0)",
                    xy=(1.0, y[int(np.argmin(measured))]),
                    xytext=(1.6, y.max() - 0.004), fontsize=7,
                    arrowprops={"arrowstyle": "->", "lw": 0.8})
    for _, row in offline.iterrows():
        m = str(row["method"])
        ax.plot(float(row["latency_median_ms"]), float(row["der"]), "D", ms=9,
                color=COLORS.get(m, "k"), label=m)
    ax.set_xscale("log")
    ax.set_xlabel("measured median emission delay (ms, log)")
    ax.set_ylabel("DER")
    ax.set_title("DER versus measured emission delay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, out)


def der_decomposition(
    summary: pd.DataFrame, out: str | Path = FIG_DIR / "der_decomposition.png"
) -> Path:
    """Stacked miss / false-alarm / confusion per variant, with the overlap floor marked.

    The floor line matters: every system here emits at most one speaker per
    window, so the overlap fraction is an irreducible miss shared by all of them.
    Without the line a reader would read that shared floor as a property of the
    online method.
    """
    df = summary.copy()
    order = [v for v in ["online", "naive_online", "offline_ahc", "offline_spectral",
                         "online_oracle_count", "offline_ahc_oracle_count",
                         "online_oracle_vad", "offline_ahc_oracle_vad"]
             if v in set(df["variant"])]
    df = df.set_index("variant").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    idx = np.arange(len(df))
    miss = df["miss"].to_numpy(dtype=float)
    fa = df["false_alarm"].to_numpy(dtype=float)
    conf = df["confusion"].to_numpy(dtype=float)
    ax.bar(idx, miss, label="miss", color="#4c78a8")
    ax.bar(idx, fa, bottom=miss, label="false alarm", color="#f58518")
    ax.bar(idx, conf, bottom=miss + fa, label="speaker confusion", color="#e45756")
    if "der_noise_scale" in df.columns:
        ax.errorbar(idx, miss + fa + conf,
                    yerr=np.nan_to_num(df["der_noise_scale"].to_numpy(dtype=float)),
                    fmt="none", ecolor="k", capsize=4, lw=1.2,
                    label=r"$\pm\sqrt{2}\,\sigma_{\rm seed}$")
    if "overlap_miss_floor" in df.columns:
        floor = float(np.nanmean(df["overlap_miss_floor"]))
        ax.axhline(floor, ls=":", color="k", lw=1.5,
                   label=f"overlap miss floor ({floor:.3f}, shared by all)")
    ax.set_xticks(idx)
    ax.set_xticklabels(df["variant"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("DER contribution")
    ax.set_title("DER decomposition by variant (oracle rows are diagnostics, not systems)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, out)


def ablation_deltas(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    out: str | Path = FIG_DIR / "ablations.png",
) -> Path:
    """Per-ablation DER change from the full system, against the noise scale.

    The grey band is ``+/- sqrt(2) * sd``. A bar inside the band contributed
    nothing measurable, whatever its p-value — which is the honest way to read
    an ablation table and is why the band is drawn rather than described.
    """
    sub = tests[tests["metric"] == "der"].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no DER ablation tests", ha="center")
        return _save(fig, out)
    # delta is full - ablation, so flip the sign to read "cost of removing".
    sub["cost"] = -sub["delta"].to_numpy(dtype=float)
    sub = sub.sort_values("cost")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    idx = np.arange(len(sub))
    noise = np.nan_to_num(sub["noise_scale"].to_numpy(dtype=float), nan=0.0)
    colors = ["#4c78a8" if v in ("survives", "suggestive") else "#bbbbbb"
              for v in sub["verdict"]]
    ax.barh(idx, sub["cost"], color=colors)
    ax.axvspan(-float(np.nanmax(noise)), float(np.nanmax(noise)), color="k", alpha=0.10,
               label=r"$\pm\sqrt{2}\,\sigma_{\rm seed}$ (run-to-run noise)")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(idx)
    ax.set_yticklabels(
        [f"{r.name_b}  [{r.verdict}]" for r in sub.itertuples()], fontsize=8
    )
    ax.set_xlabel("DER change when the mechanism is removed (positive = removal hurts)")
    ax.set_title("Ablations, one config switch each")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out)


def calibration_panels(
    calib: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    out: str | Path = FIG_DIR / "calibration.png",
) -> Path:
    """Reliability, risk-coverage, and the abstention operating points.

    The risk-coverage panel is the practically useful one: it says what error rate
    survives if the least-confident fraction of turns is deferred to an offline
    pass. The oracle curve is the achievable floor, not a target.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    for method, sub in calib.groupby("method", sort=False):
        acc = float(np.nanmean(sub["raw_accuracy"]))
        conf = float(np.nanmean(sub["raw_mean_confidence"]))
        ax.plot([conf], [acc], "o", ms=11, color=COLORS.get(str(method), "k"), label=str(method))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("turn-assignment accuracy")
    ax.set_title("Confidence versus accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for method, sub in risk_coverage.groupby("method", sort=False):
        grid = np.linspace(0.02, 1.0, 60)
        per_seed = []
        for _, s in sub.groupby("seed"):
            s = s.sort_values("coverage")
            per_seed.append(np.interp(grid, s["coverage"], s["risk"]))
        if per_seed:
            ax.plot(grid, np.mean(per_seed, axis=0), lw=2,
                    color=COLORS.get(str(method), "k"), label=str(method))
    ax.set_xlabel("coverage (fraction of turn decisions kept)")
    ax.set_ylabel("error rate over kept decisions")
    ax.set_title("Risk-coverage: deferring the uncertain turns")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    cols = [c for c in calib.columns if c.startswith("risk_at_coverage_")]
    cols = sorted(cols, key=lambda c: -int(c.rsplit("_", 1)[1]))
    labels = [c.rsplit("_", 1)[1] + "%" for c in cols]
    width = 0.8 / max(1, calib["method"].nunique())
    for i, (method, sub) in enumerate(calib.groupby("method", sort=False)):
        vals = [float(np.nanmean(sub[c])) for c in cols]
        ax.bar(np.arange(len(cols)) + i * width, vals, width,
               color=COLORS.get(str(method), "k"), label=str(method))
    ax.set_xticks(np.arange(len(cols)) + 0.4 - width / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("coverage")
    ax.set_ylabel("error rate over kept decisions")
    ax.set_title("Abstention operating points")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    return _save(fig, out)


def efficiency_panels(
    efficiency: pd.DataFrame,
    scaling: pd.DataFrame,
    out: str | Path = FIG_DIR / "efficiency.png",
) -> Path:
    """Emission latency, clustering wall-clock scaling, and per-window clustering cost.

    The middle panel is the compute claim: online clustering should be linear in
    the number of windows while the offline methods are super-linear. The right
    panel divides by window count, so a flat line is O(1) per window and a rising
    line is not.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    ax = axes[0]
    idx = np.arange(len(efficiency))
    ax.bar(idx, efficiency["emission_latency_median_ms"],
           color=[COLORS.get(m, "k") for m in efficiency["method"]])
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(efficiency["method"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("median emission delay (ms, log)")
    ax.set_title("Emission delay per turn decision")
    ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1]
    for method, sub in scaling.groupby("method", sort=False):
        sub = sub.sort_values("n_speech_windows")
        ax.plot(sub["n_speech_windows"], sub["cluster_median_ms"], "o-", lw=2,
                color=COLORS.get(str(method), "k"), label=str(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("speech windows in the recording (log)")
    ax.set_ylabel("clustering wall-clock (ms, log)")
    ax.set_title("Clustering cost versus recording length")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    for method, sub in scaling.groupby("method", sort=False):
        sub = sub.sort_values("n_speech_windows")
        ax.plot(sub["n_speech_windows"], sub["ms_per_window"], "o-", lw=2,
                color=COLORS.get(str(method), "k"), label=str(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("speech windows (log)")
    ax.set_ylabel("clustering ms per window (log)")
    ax.set_title("Per-window cost: flat is O(1), rising is not")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out)


def data_overview(recording, out: str | Path = FIG_DIR / "data_overview.png") -> Path:
    """One generated conversation: features on top, the exact reference below.

    Included because the "ground truth by construction" claim is much easier to
    believe when you can see the turn boundaries line up with the energy.
    """
    ref = recording.reference_matrix()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    im = ax.imshow(recording.features.T, aspect="auto", origin="lower",
                   cmap="magma", interpolation="nearest",
                   extent=(0.0, recording.duration_s, 0, recording.features.shape[1]))
    ax.set_ylabel("feature channel")
    ax.set_title(
        f"{recording.name}: {recording.n_speakers} speakers, "
        f"{len(recording.turns)} turns, "
        f"{recording.overlap_fraction():.1%} of speech overlapped"
    )
    fig.colorbar(im, ax=ax, pad=0.01, label="log power")

    ax = axes[1]
    t = np.arange(recording.n_frames) / recording.frame_rate
    for s in range(recording.n_speakers):
        ax.fill_between(t, s, s + 0.8, where=ref[:, s], step="mid",
                        color=plt.cm.tab10(s % 10), alpha=0.85)
    ax.set_yticks(np.arange(recording.n_speakers) + 0.4)
    ax.set_yticklabels([f"spk {s}" for s in range(recording.n_speakers)])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("reference")
    ax.set_xlim(0, recording.duration_s)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out)


def similarity_histogram(
    same: np.ndarray, diff: np.ndarray,
    out: str | Path = FIG_DIR / "embedding_separability.png",
) -> Path:
    """Same-speaker versus different-speaker window cosine similarity.

    This is the figure that explains every threshold in the config, and it is why
    the first version of this project mis-set ``spawn_threshold`` to 0.55: the
    embedding space is a narrow cone, not an isotropic ball, so the useful
    operating point sits near 0.85.
    """
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bins = np.linspace(-1, 1, 80)
    ax.hist(diff, bins=bins, alpha=0.6, density=True, label="different speakers",
            color="#e45756")
    ax.hist(same, bins=bins, alpha=0.6, density=True, label="same speaker",
            color="#4c78a8")
    for v, name, color in [(np.median(same), "same median", "#4c78a8"),
                           (np.median(diff), "different median", "#e45756")]:
        ax.axvline(float(v), color=color, ls="--", lw=1.6,
                   label=f"{name} = {float(v):.3f}")
    ax.set_xlabel("cosine similarity between window embeddings")
    ax.set_ylabel("density")
    ax.set_title("Why the thresholds are near 0.85, not 0.55")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out)
