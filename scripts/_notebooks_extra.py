"""Notebooks 03-05, kept in a separate module so ``make_notebooks.py`` stays readable.

Imported opportunistically by ``make_notebooks.py``; if this file is absent only
notebooks 01 and 02 are generated.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

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


def nb03() -> nbformat.NotebookNode:
    return _nb([
        new_markdown_cell(
            "# 03 · Ablations\n\n"
            "One config switch each, so each row attributes something. Every delta "
            "is read against `sqrt(2) * seed_sd` — the run-to-run scale of a "
            "difference between two runs. A bar inside that band contributed "
            "nothing measurable, whatever its p-value."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            "summary = table('ablation_summary.csv')\n"
            "cols = ['ablation', 'der', 'der_seed_sd', 'der_noise_scale', 'confusion',\n"
            "        'n_speakers_pred', 'extra_mean_query_size']\n"
            "print(summary[[c for c in cols if c in summary.columns]].round(4)\n"
            "      .to_string(index=False))"
        ),
        new_markdown_cell(
            "`extra_mean_query_size` is the diagnostic that shows the bounded-window "
            "mechanism is actually engaged: it is the number of windows averaged "
            "into each decision's query. It must be 1.00 exactly when "
            "`bounded_window: false`, and above 1 otherwise. If it were 1.00 for the "
            "full system, the ablation would be testing nothing."
        ),
        new_code_cell(
            "tests = table('ablation_tests.csv')\n"
            "sub = tests[tests.metric == 'der']\n"
            "print(sub[['name_b', 'mean_a', 'mean_b', 'delta', 'ci_lower', 'ci_upper',\n"
            "           'p_adjusted', 'noise_scale', 'noise_ratio', 'verdict']]\n"
            "      .round(5).to_string(index=False))"
        ),
        new_code_cell(
            "sub = sub.copy()\n"
            "sub['cost'] = -sub['delta']\n"
            "sub = sub.sort_values('cost')\n"
            "noise = float(np.nanmax(sub['noise_scale']))\n"
            "plt.figure(figsize=(9, 4))\n"
            "colors = ['#4c78a8' if v in ('survives', 'suggestive') else '#bbbbbb'\n"
            "          for v in sub.verdict]\n"
            "plt.barh(np.arange(len(sub)), sub['cost'], color=colors)\n"
            "plt.axvspan(-noise, noise, color='k', alpha=0.10,\n"
            "            label=r'$\\pm\\sqrt{2}\\sigma_{seed}$')\n"
            "plt.axvline(0, color='k', lw=1)\n"
            "plt.yticks(np.arange(len(sub)),\n"
            "           [f'{r.name_b}  [{r.verdict}]' for r in sub.itertuples()],\n"
            "           fontsize=8)\n"
            "plt.xlabel('DER change when the mechanism is removed')\n"
            "plt.grid(alpha=0.3, axis='x'); plt.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "### The non-causal row is not a system\n\n"
            "`noncausal` uses centred convolution padding, so each frame's "
            "representation depends on roughly 150 ms of *future* audio. Its "
            "reported emission latency therefore understates its true latency by "
            "that amount, and it is a diagnostic rather than a deployable "
            "configuration. `tests/test_causality.py` requires it to **fail** the "
            "bit-identity check the shipped model passes."
        ),
    ])


def nb04() -> nbformat.NotebookNode:
    return _nb([
        new_markdown_cell(
            "# 04 · Calibration and abstention\n\n"
            "A bounded-latency diarizer is less accurate than an offline one. If it "
            "also knows *which* of its turn decisions are probably wrong, a "
            "deployment can route that minority to a slower offline pass and buy "
            "most of the gap back. This notebook measures whether it does."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            "cal = table('calibration.csv')\n"
            "cols = ['method', 'seed', 'temperature', 'raw_accuracy', 'raw_mean_confidence',\n"
            "        'raw_overconfidence', 'raw_ece', 'cal_ece', 'raw_ace', 'raw_brier',\n"
            "        'raw_error_detection_auroc', 'raw_aurc', 'raw_aurc_oracle']\n"
            "print(cal[[c for c in cols if c in cal.columns]].round(4).to_string(index=False))"
        ),
        new_markdown_cell(
            "`temperature` is fitted on the **dev** split only, never on test. "
            "`cal_ece` is the ECE after that single scalar recalibration.\n\n"
            "Note the asymmetry that has to be stated: for the online system the "
            "confidence is the assignment margin *available at decision time*. For "
            "the offline system there was no decision time, so its confidence is a "
            "**post hoc** similarity to the final cluster centroid. The two are not "
            "the same kind of quantity."
        ),
        new_code_cell(
            "rc = table('risk_coverage.csv')\n"
            "plt.figure(figsize=(11, 4.2))\n"
            "plt.subplot(1, 2, 1)\n"
            "for method, sub in rc.groupby('method'):\n"
            "    grid = np.linspace(0.02, 1.0, 60)\n"
            "    curves = [np.interp(grid, s.sort_values('coverage').coverage,\n"
            "                        s.sort_values('coverage').risk)\n"
            "              for _, s in sub.groupby('seed')]\n"
            "    plt.plot(grid, np.mean(curves, axis=0), lw=2, label=method)\n"
            "plt.xlabel('coverage (fraction of turn decisions kept)')\n"
            "plt.ylabel('error rate over kept decisions')\n"
            "plt.title('Risk-coverage'); plt.grid(alpha=0.3); plt.legend(fontsize=8)\n"
            "plt.subplot(1, 2, 2)\n"
            "cols = [c for c in cal.columns if c.startswith('risk_at_coverage_')]\n"
            "cols = sorted(cols, key=lambda c: -int(c.rsplit('_', 1)[1]))\n"
            "w = 0.8 / cal.method.nunique()\n"
            "for i, (method, sub) in enumerate(cal.groupby('method')):\n"
            "    plt.bar(np.arange(len(cols)) + i * w,\n"
            "            [float(np.nanmean(sub[c])) for c in cols], w, label=method)\n"
            "plt.xticks(np.arange(len(cols)) + 0.4 - w / 2,\n"
            "           [c.rsplit('_', 1)[1] + '%' for c in cols])\n"
            "plt.xlabel('coverage'); plt.ylabel('error rate over kept decisions')\n"
            "plt.title('Abstention operating points'); plt.grid(alpha=0.3, axis='y')\n"
            "plt.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "### Reading the AUROC honestly\n\n"
            "`raw_error_detection_auroc` is the probability that a randomly chosen "
            "*correct* decision is more confident than a randomly chosen *wrong* "
            "one. 0.5 is useless. Compare `raw_aurc` against `raw_aurc_oracle`: the "
            "oracle abstains on exactly the errors and is the achievable floor, not "
            "a target."
        ),
        new_code_cell(
            "summ = table('calibration_summary.csv')\n"
            "keep = ['method', 'raw_accuracy', 'raw_ece', 'raw_ece_noise_scale',\n"
            "        'cal_ece', 'raw_error_detection_auroc',\n"
            "        'raw_error_detection_auroc_noise_scale', 'raw_aurc', 'raw_aurc_oracle']\n"
            "print(summ[[c for c in keep if c in summ.columns]].round(4).to_string(index=False))"
        ),
    ])


def nb05() -> nbformat.NotebookNode:
    return _nb([
        new_markdown_cell(
            "# 05 · Full scale on a GPU (Colab / Kaggle)\n\n"
            "The same code, at a scale this 4-core laptop cannot reach, plus the "
            "optional real-data path.\n\n"
            "**This notebook is not executed in CI and its outputs are not "
            "committed** — it needs a GPU runtime and, for the real-data section, a "
            "dataset the repository deliberately does not download. Everything "
            "shipped in `results/` comes from the CPU path in notebooks 01-04."
        ),
        new_markdown_cell(
            "## Setup\n\n"
            "Uncomment on a fresh Colab runtime. The dependency set is deliberately "
            "small: torch, numpy, pandas, pyyaml, matplotlib. No pyannote, no "
            "speechbrain, no scipy."
        ),
        new_code_cell(
            "# !git clone https://github.com/arslan-ahmad/streaming-speaker-diarization.git\n"
            "# %cd streaming-speaker-diarization\n"
            "# !pip install -q torch numpy pandas pyyaml matplotlib tqdm\n"
            "import sys, os\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))\n"
            "import torch\n"
            "print('cuda available:', torch.cuda.is_available())"
        ),
        new_markdown_cell(
            "## Scale up\n\n"
            "The shipped config is sized for a CPU budget: 900 training steps, 24 "
            "test recordings of 60 s, a 40-channel front end. On a GPU all of those "
            "can grow by an order of magnitude with no code change — every knob is a "
            "config field, and `--set` overrides are typed by the target field so a "
            "bool stays a bool."
        ),
        new_code_cell(
            "from streamdiar.config import load_config\n"
            "cfg = load_config('configs/base.yaml', [\n"
            "    'train.steps=9000',\n"
            "    'train.n_speakers_per_batch=48',\n"
            "    'train.threads=8',\n"
            "    'embedder.channels=192',\n"
            "    'embedder.dilations=1,2,4,8,16,32',\n"
            "    'embedder.embed_dim=128',\n"
            "    'data.n_features=64',\n"
            "    'data.duration_s=300',\n"
            "    'data.n_train_speakers=1200',\n"
            "    'data.n_test_recordings=100',\n"
            "])\n"
            "print('receptive field will be',\n"
            "      1 + 2 * sum(cfg.embedder.dilations), 'frames')\n"
            "print('test audio:',\n"
            "      cfg.data.n_test_recordings * cfg.data.duration_s / 3600, 'hours')"
        ),
        new_code_cell(
            "# from streamdiar.engine import train_embedder, fit_vad\n"
            "# from streamdiar.data.generator import generate_split\n"
            "# model = train_embedder(cfg, seed=0, progress=True)\n"
            "# fit_vad(model, cfg, generate_split(cfg.data, 0, 'dev'))"
        ),
        new_markdown_cell(
            "## The real-data path\n\n"
            "`streamdiar.data.real` turns a WAV plus its RTTM reference into the same "
            "`Recording` object the generator produces, through a hand-rolled log-mel "
            "front end (25 ms Hann frames, 10 ms hop, 40 triangular mel filters, "
            "**natural** log so the units match the generator exactly). Every "
            "diarizer and every metric then works unchanged.\n\n"
            "The caveat that must travel with any number produced this way: an RTTM "
            "reference is a human annotation with boundary error of tens of "
            "milliseconds, which is the same order as the differences the latency "
            "sweep resolves. That is precisely why the shipped results use generated "
            "data with an exact reference."
        ),
        new_code_cell(
            "# from streamdiar.data.real import load_real_split\n"
            "# from streamdiar.engine import evaluate, aggregate\n"
            "# recs = load_real_split('data/ami', frame_rate=100, n_mels=cfg.data.n_features)\n"
            "# for method in ['online', 'offline_ahc', 'offline_spectral']:\n"
            "#     agg = aggregate(evaluate(model, cfg, recs, method=method))\n"
            "#     print(f\"{method:18s} DER {agg['der']:.4f}  \"\n"
            "#           f\"latency {agg['latency_median_ms']:.0f} ms\")\n"
            "print('real-data cells are commented out: no dataset is bundled')"
        ),
        new_markdown_cell(
            "## The latency sweep at scale\n\n"
            "The interesting question a GPU run can answer that this laptop cannot: "
            "does the optimum of the DER-versus-latency curve move when the embedder "
            "is strong enough that single-window embeddings are already reliable? "
            "The mechanism's benefit is variance reduction on the query, so a "
            "lower-variance embedder should need *less* lookahead — the optimum "
            "should move left. That is a prediction, not a result: it has not been "
            "run here, and it is listed as open work in `docs/RESULTS.md`."
        ),
        new_code_cell(
            "# from streamdiar.pipelines.latency import run_latency_sweep, summarise_curve\n"
            "# tbl = run_latency_sweep(cfg, seeds=(0, 1, 2))\n"
            "# print(summarise_curve(tbl).to_string(index=False))"
        ),
    ])


EXTRA_BUILDERS = {
    "03_ablations": nb03,
    "04_calibration_and_abstention": nb04,
    "05_colab_full_scale": nb05,
}
