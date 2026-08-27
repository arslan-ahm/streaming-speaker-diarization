"""Generate the notebooks programmatically with nbformat, then execute them.

    uv run python scripts/make_notebooks.py            # write only
    uv run python scripts/make_notebooks.py --execute   # write and run with outputs

Generated rather than hand-written so that every cell has a stable id and the
notebooks cannot drift from the code they document. They read the **committed
CSVs** wherever possible rather than recomputing, so executing them is fast and
their numbers cannot disagree with the tables.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

PREAMBLE = """\
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch; torch.set_num_threads(2)

TABLES = os.path.join("..", "results", "tables")
def table(name):
    return pd.read_csv(os.path.join(TABLES, name))
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 60)
"""


def _nb(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:03d}"
    return nb


def nb01() -> nbformat.NotebookNode:
    return _nb([
        new_markdown_cell(
            "# 01 · The data and the core idea\n\n"
            "The whole project rests on one property of this dataset: **the "
            "reference is exact by construction**. Turns are sampled first, "
            "features are rendered from them, so there is no annotation error "
            "anywhere in the evaluation. This notebook shows that, then shows "
            "why the embedder has something to learn."
        ),
        new_code_cell(PREAMBLE),
        new_markdown_cell(
            "## A generated conversation\n\n"
            "Features on top, the exact reference below. The turn boundaries are "
            "where the generator put them, not where an annotator guessed."
        ),
        new_code_cell(
            "from streamdiar.config import load_config\n"
            "from streamdiar.data.generator import (\n"
            "    dataset_stats, generate_recording, generate_split,\n"
            ")\n"
            "cfg = load_config('../configs/base.yaml')\n"
            "rec = generate_recording(cfg.data, 0, 1000, 'test')\n"
            "print(rec.name, rec.features.shape,\n"
            "      f'{rec.n_speakers} speakers, {len(rec.turns)} turns')\n"
            "print(f'overlap {rec.overlap_fraction():.1%} of speech frames, '\n"
            "      f'speech {rec.speech_fraction():.1%} of all frames')"
        ),
        new_code_cell(
            "ref = rec.reference_matrix()\n"
            "fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,\n"
            "                         gridspec_kw={'height_ratios': [3, 1]})\n"
            "im = axes[0].imshow(rec.features.T, aspect='auto', origin='lower', cmap='magma',\n"
            "                    extent=(0, rec.duration_s, 0, rec.features.shape[1]))\n"
            "axes[0].set_ylabel('feature channel'); axes[0].set_title(rec.name)\n"
            "fig.colorbar(im, ax=axes[0], pad=0.01, label='log power')\n"
            "t = np.arange(rec.n_frames) / rec.frame_rate\n"
            "for s in range(rec.n_speakers):\n"
            "    axes[1].fill_between(t, s, s + 0.8, where=ref[:, s], step='mid',\n"
            "                         color=plt.cm.tab10(s % 10), alpha=0.85)\n"
            "axes[1].set_yticks(np.arange(rec.n_speakers) + 0.4)\n"
            "axes[1].set_yticklabels([f'spk {s}' for s in range(rec.n_speakers)])\n"
            "axes[1].set_xlabel('time (s)'); axes[1].set_ylabel('reference')\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Overlap really is summed in the power domain\n\n"
            "Two people talking at once combine as `log(exp(a) + exp(b))`, not "
            "`a + b`. If it were the latter, overlap would be a trivially "
            "detectable amplitude spike and the overlap experiments would be "
            "meaningless. Doubling the power adds only `log 2 ≈ 0.69`."
        ),
        new_code_cell(
            "counts = ref.sum(axis=1)\n"
            "rows = []\n"
            "for k, name in [(0, 'silence'), (1, 'one speaker'), (2, 'two speakers')]:\n"
            "    sel = counts == k if k < 2 else counts >= 2\n"
            "    if sel.any():\n"
            "        rows.append({'region': name, 'frames': int(sel.sum()),\n"
            "                     'mean log power': float(rec.features[sel].mean())})\n"
            "disp = pd.DataFrame(rows)\n"
            "print(disp.to_string(index=False))\n"
            "print('\\nlog(2) =', round(float(np.log(2)), 4),\n"
            "      '<- the most a second equal-power talker can add')"
        ),
        new_markdown_cell(
            "## Split statistics\n\n"
            "The test split is 24 recordings of 60 s, 2–5 speakers each, drawn "
            "from a speaker pool disjoint from training."
        ),
        new_code_cell(
            "stats = dataset_stats(generate_split(cfg.data, 0, 'test'))\n"
            "print(pd.Series(stats).to_string())"
        ),
        new_markdown_cell(
            "## Why the thresholds are near 0.85, not 0.55\n\n"
            "This is the figure that explains every threshold in the config — and "
            "the mistake that motivated measuring it. The embedding space is a "
            "narrow cone, so a cosine-similarity intuition of \"0.55 means "
            "different speakers\" is badly wrong. Guessing 0.55 made the diarizer "
            "find 1.62 speakers where there were 3.50."
        ),
        new_code_cell(
            "from streamdiar.pipelines.common import dev_split, load_or_train\n"
            "from streamdiar.pipelines.tuning import precompute, similarity_distributions\n"
            "os.chdir('..')  # checkpoints/ and results/ are repo-relative\n"
            "model = load_or_train(cfg, 0, verbose=False)\n"
            "cached = precompute(model, cfg, dev_split(cfg, 0))\n"
            "same, diff = similarity_distributions(cached, cfg.frames_per_hop)\n"
            "os.chdir('notebooks')\n"
            "bins = np.linspace(-1, 1, 80)\n"
            "plt.figure(figsize=(9, 4))\n"
            "plt.hist(diff, bins=bins, alpha=0.6, density=True,\n"
            "         label='different speakers', color='#e45756')\n"
            "plt.hist(same, bins=bins, alpha=0.6, density=True,\n"
            "         label='same speaker', color='#4c78a8')\n"
            "plt.axvline(np.median(same), color='#4c78a8', ls='--',\n"
            "         label=f'same median {np.median(same):.3f}')\n"
            "plt.axvline(np.median(diff), color='#e45756', ls='--',\n"
            "         label=f'diff median {np.median(diff):.3f}')\n"
            "plt.axvline(0.83, color='k', ls=':', lw=2, label='tuned spawn_threshold 0.83')\n"
            "plt.xlabel('cosine similarity between window embeddings'); plt.ylabel('density')\n"
            "plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Causality, as a guarantee rather than a statistic\n\n"
            "Perturb every frame after `t0` and the embeddings of all windows "
            "ending at or before `t0` are **bit-identical**. Not close — equal. "
            "The non-causal variant fails the same check, which is what makes the "
            "clean result meaningful."
        ),
        new_code_cell(
            "from streamdiar.config import replace\n"
            "from streamdiar.models.embedder import build_embedder\n"
            "x = np.random.RandomState(0).randn(2000, cfg.data.n_features).astype(np.float32)\n"
            "x2 = x.copy(); t0 = 1200\n"
            "x2[t0:] = np.random.RandomState(1).randn(2000 - t0, cfg.data.n_features)\n"
            "rows = []\n"
            "for causal in [True, False]:\n"
            "    m = build_embedder(replace(cfg.embedder, causal=causal), cfg.data.n_features,\n"
            "                       model.feature_mean, model.feature_sd, seed=0)\n"
            "    a, _, ends = m.embed_recording(x, cfg.frames_per_window, cfg.frames_per_hop)\n"
            "    b, _, _ = m.embed_recording(x2, cfg.frames_per_window, cfg.frames_per_hop)\n"
            "    early = ends <= t0\n"
            "    rows.append({'causal': causal, 'windows checked': int(early.sum()),\n"
            "                 'max abs difference': float(np.abs(a[early] - b[early]).max())})\n"
            "print(pd.DataFrame(rows).to_string(index=False))"
        ),
    ])


def nb02() -> nbformat.NotebookNode:
    return _nb([
        new_markdown_cell(
            "# 02 · The latency/accuracy tradeoff\n\n"
            "The headline result. The offline reference is **not a competitor** — "
            "it is the infinite-latency asymptote. What is being measured is the "
            "cost of a bounded emission delay, and where that cost stops mattering."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            "curve = table('latency_curve.csv')\n"
            "online = curve[(curve.method == 'online') & np.isfinite(curve.latency_budget_ms)]\n"
            "online = online.sort_values('latency_budget_ms')\n"
            "cols = ['latency_budget_ms', 'der', 'der_noise_scale', 'confusion',\n"
            "        'n_speakers_pred', 'latency_median_ms', 'extra_mean_query_size']\n"
            "print(online[[c for c in cols if c in online.columns]].to_string(index=False))"
        ),
        new_markdown_cell(
            "The curve is **flat below one hop** by construction: a budget of `B` ms "
            "buys `floor(B / 250)` hops of lookahead, so 0 ms and 125 ms are the "
            "same system. Both are in the sweep so that step is visible rather than "
            "assumed away."
        ),
        new_code_cell(
            "offline = curve[curve.method.str.startswith('offline')]\n"
            "cols = ['method', 'der', 'confusion', 'latency_median_ms']\n"
            "print(offline[cols].to_string(index=False))"
        ),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))\n"
            "x = online.latency_budget_ms.to_numpy(float); y = online.der.to_numpy(float)\n"
            "b = np.nan_to_num(online.der_noise_scale.to_numpy(float))\n"
            "axes[0].plot(x, y, 'o-', lw=2, color='#1f77b4', label='online')\n"
            "axes[0].fill_between(x, y - b, y + b, alpha=0.18, color='#1f77b4',\n"
            "                     label=r'$\\pm\\sqrt{2}\\sigma_{seed}$')\n"
            "for _, r in offline.iterrows():\n"
            "    axes[0].axhline(r.der, ls='--', lw=1.5,\n"
            "                    color='#2ca02c' if 'ahc' in r.method else '#9467bd',\n"
            "                    label=f\"{r.method} (offline)\")\n"
            "axes[0].set_xscale('symlog', linthresh=125)\n"
            "axes[0].set_xlabel('latency budget (ms)'); axes[0].set_ylabel('DER')\n"
            "axes[0].set_title('DER vs latency budget'); axes[0].grid(alpha=0.3)\n"
            "axes[0].legend(fontsize=7.5)\n"
            "axes[1].plot(np.maximum(online.latency_median_ms, 1),\n"
            "         y, 'o-', lw=2, color='#1f77b4')\n"
            "for _, r in offline.iterrows():\n"
            "    axes[1].plot(r.latency_median_ms, r.der, 'D', ms=10,\n"
            "                 color='#2ca02c' if 'ahc' in r.method else '#9467bd',\n"
            "                 label=r.method)\n"
            "axes[1].set_xscale('log'); axes[1].set_xlabel('measured median emission delay (ms)')\n"
            "axes[1].set_ylabel('DER')\n"
            "axes[1].set_title('...vs what it actually costs in delay')\n"
            "axes[1].grid(alpha=0.3, which='both'); axes[1].legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Where is the gap, and is it real?\n\n"
            "Every difference is placed against `sqrt(2) * seed_sd` — the "
            "run-to-run scale of a difference between two runs. Anything smaller "
            "is noise regardless of its p-value."
        ),
        new_code_cell(
            "summary = table('method_summary.csv')\n"
            "cols = ['variant', 'der', 'der_seed_sd', 'der_noise_scale', 'miss',\n"
            "        'false_alarm', 'confusion', 'jer', 'latency_median_ms', 'n_speakers_pred']\n"
            "print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))"
        ),
        new_code_cell(
            "tests = table('statistical_tests.csv')\n"
            "sub = tests[tests.metric == 'der']\n"
            "print(sub[['name_b', 'mean_a', 'mean_b', 'delta', 'ci_lower', 'ci_upper',\n"
            "           'p_value', 'p_adjusted', 'noise_scale', 'noise_ratio',\n"
            "           'verdict']].to_string(index=False))"
        ),
        new_markdown_cell(
            "**Unit of analysis.** These paired tests are over the 24 test "
            "recordings of one seed, so they condition on *one trained model per "
            "method*. They say whether these two sets of weights differ on this "
            "test set — not whether the method is better. That is why the "
            "`noise_ratio` column, not `p_adjusted`, decides the verdict."
        ),
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", default=None, help="Comma-separated notebook stems.")
    args = ap.parse_args(argv)

    NB_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_data_and_the_core_idea": nb01,
        "02_latency_tradeoff": nb02,
    }
    try:
        from _notebooks_extra import EXTRA_BUILDERS  # type: ignore

        builders.update(EXTRA_BUILDERS)
    except Exception:
        pass

    wanted = set(args.only.split(",")) if args.only else set(builders)
    written = []
    for stem, build in builders.items():
        if stem not in wanted:
            continue
        path = NB_DIR / f"{stem}.ipynb"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            nbformat.write(build(), fh)
        written.append(path)
        print(f"wrote {path}")

    if args.execute:
        for path in written:
            print(f"executing {path} ...", flush=True)
            cmd = [
                sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", f"--ExecutePreprocessor.timeout={args.timeout}", str(path),
            ]
            res = subprocess.run(cmd, cwd=NB_DIR, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:])
                print(res.stderr[-3000:])
                return res.returncode
            print(f"  ok {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
