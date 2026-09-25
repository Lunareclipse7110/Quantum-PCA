"""
generate_report.py
====================
Team 4 — Quantum Principal Component Analysis
Turns results/results.csv + results/figures/*.png into ONE self-contained
HTML file (charts embedded as base64, no external files needed) that opens
in any browser as a readable report/dashboard — something to actually show
your professor, instead of a CSV + a folder of PNGs.

Usage
-----
    python -m src.generate_report

Output: results/Team4_Quantum_PCA_Report.html
(open it by double-clicking, or File > Open in any browser; File > Print >
Save as PDF from the browser gives you a hard copy / hand-in file too)
"""
import base64
import os
from datetime import datetime

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
OUT_PATH = os.path.join(RESULTS_DIR, "Team4_Quantum_PCA_Report.html")

EXPERIMENT_ROWS = [
    ("E1", 2, 2, "ideal"), ("E2", 2, 4, "ideal"),
    ("E3", 4, 2, "ideal"), ("E4", 4, 4, "ideal"),
    ("E5", 4, 4, "low"), ("E6", 4, 4, "moderate"),
    ("E7", 6, 4, "ideal"), ("E8", 6, 4, "moderate"),
]

FIGURE_SPECS = [
    ("reconstruction_quality", "Reconstruction quality: quantum vs classical PCA across all 8 experiments."),
    ("noise_sensitivity", "Reconstruction error vs noise level, qubits and depth held fixed (E4/E5/E6)."),
    ("resource_usage", "Transpiled circuit depth and total gate count per experiment."),
    ("training_curve_E4", "Training convergence (cost vs. COBYLA iteration) for experiment E4."),
    ("measurement_histogram_E4_vs_E6", "SWAP-test ancilla measurement counts, ideal vs. moderate noise."),
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_and_aggregate():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found. Run `python main.py` (or `python main.py --quick`) first."
        )
    df = pd.read_csv(RESULTS_CSV)
    group_cols = ["dataset", "exp_id", "n_qubits", "depth", "noise_level"]
    metrics = [
        "q_reconstruction_error", "q_mean_fidelity", "q_circuit_depth", "q_n_gates_total",
        "q_train_time_s", "q_eval_time_s",
        "c_reconstruction_mse", "c_explained_variance_ratio_sum", "c_fit_time_s",
    ]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.values]
    return df, agg


def headline_stats(df, agg):
    ideal = agg[agg["noise_level"] == "ideal"]
    moderate = agg[agg["noise_level"] == "moderate"]
    stats = {
        "n_runs": len(df),
        "n_datasets": df["dataset"].nunique(),
        "qubit_range": f"{df['n_qubits'].min()}\u2013{df['n_qubits'].max()}",
        "q_err_ideal": ideal["q_reconstruction_error_mean"].mean(),
        "c_err_ideal": ideal["c_reconstruction_mse_mean"].mean(),
    }
    if len(moderate) and len(ideal):
        stats["noise_penalty_pct"] = (
            (moderate["q_reconstruction_error_mean"].mean() - ideal["q_reconstruction_error_mean"].mean())
            / max(ideal["q_reconstruction_error_mean"].mean(), 1e-9) * 100
        )
    else:
        stats["noise_penalty_pct"] = float("nan")
    return stats


def fig_to_data_uri(name: str) -> str:
    path = os.path.join(FIG_DIR, f"{name}.png")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------- #
# HTML building blocks
# --------------------------------------------------------------------------- #
def esc(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def build_results_table(agg, dataset: str) -> str:
    sub = agg[agg["dataset"] == dataset].sort_values("exp_id")
    cols = [
        ("exp_id", "Exp"), ("n_qubits", "Qubits"), ("depth", "Depth"), ("noise_level", "Noise"),
        ("q_reconstruction_error_mean", "Quantum recon. error"),
        ("c_reconstruction_mse_mean", "Classical recon. MSE"),
        ("q_circuit_depth_mean", "Circuit depth"),
        ("q_n_gates_total_mean", "Gate count"),
    ]
    head = "".join(f"<th>{label}</th>" for _, label in cols)
    rows = []
    for _, row in sub.iterrows():
        cells = "".join(
            f"<td class='num'>{esc(row[key])}</td>" if key not in ("exp_id", "n_qubits", "depth", "noise_level")
            else f"<td>{esc(row[key])}</td>"
            for key, _ in cols
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<table class='data-table'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_matrix_table() -> str:
    head = "<th>Exp</th><th>Qubits</th><th>Depth</th><th>Noise</th>"
    rows = "".join(
        f"<tr><td>{eid}</td><td class='num'>{q}</td><td class='num'>{d}</td><td>{noise}</td></tr>"
        for eid, q, d, noise in EXPERIMENT_ROWS
    )
    return f"<table class='data-table small'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def build_figures_grid(dataset: str) -> str:
    cards = []
    for slug, caption in FIGURE_SPECS:
        uri = fig_to_data_uri(f"{dataset}_{slug}")
        if not uri:
            continue
        cards.append(
            f"<figure class='fig-card'><img src='{uri}' alt='{caption}'>"
            f"<figcaption>{caption}</figcaption></figure>"
        )
    return f"<div class='fig-grid'>{''.join(cards)}</div>"


def stat_block(value: str, label: str, tone: str = "accent") -> str:
    return f"<div class='stat stat-{tone}'><div class='stat-value'>{value}</div><div class='stat-label'>{label}</div></div>"


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #
def build_html(df, agg) -> str:
    stats = headline_stats(df, agg)
    datasets = sorted(df["dataset"].unique())
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    overview_stats = "".join([
        stat_block(str(stats["n_runs"]), "experiment runs recorded"),
        stat_block(stats["qubit_range"], "qubits tested"),
        stat_block(f"{stats['q_err_ideal']:.3f}", "mean quantum recon. error (ideal noise)"),
        stat_block(f"{stats['c_err_ideal']:.3f}", "mean classical PCA recon. error (ideal noise)", tone="classical"),
        stat_block(
            f"+{stats['noise_penalty_pct']:.0f}%" if stats["noise_penalty_pct"] == stats["noise_penalty_pct"] else "n/a",
            "reconstruction error increase, ideal \u2192 moderate noise", tone="warn",
        ),
    ])

    dataset_sections = []
    nav_dataset_links = []
    for ds in datasets:
        label = ds.upper()
        dataset_sections.append(f"""
        <section id="results-{ds}">
          <h2>Results \u2014 {label}</h2>
          <p class="lede">Quantum autoencoder vs. classical PCA, all 8 experiments, mean over repeated runs.</p>
          {build_results_table(agg, ds)}
          {build_figures_grid(ds)}
        </section>""")
        nav_dataset_links.append(f'<a href="#results-{ds}">Results: {label}</a>')

    nav_links = "\n".join([
        '<a href="#overview">Overview</a>',
        '<a href="#methodology">Methodology</a>',
        '<a href="#experiments">Experiment matrix</a>',
        *nav_dataset_links,
        '<a href="#limitations">Limitations</a>',
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Team 4 \u2014 Quantum PCA Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{CSS}
</style>
</head>
<body>
<nav class="sidebar">
  <div class="brand">
    <div class="brand-kicker">Team 4</div>
    <div class="brand-title">Quantum PCA</div>
  </div>
  <div class="nav-links">{nav_links}</div>
  <div class="sidebar-foot">Generated {generated}</div>
</nav>

<main>
  <header class="page-head">
    <h1>Quantum Principal Component Analysis</h1>
    <p class="subtitle">Quantum-autoencoder-based dimensionality reduction vs. classical PCA, evaluated across qubit count, circuit depth and simulated noise \u2014 Wine and Breast Cancer Wisconsin (WDBC) datasets.</p>
  </header>

  <section id="overview">
    <h2>Overview</h2>
    <div class="stat-row">{overview_stats}</div>
  </section>

  <section id="methodology">
    <h2>Methodology</h2>
    <p>"Quantum PCA" is implemented as a trainable <strong>quantum autoencoder</strong>: a parameterized circuit
    <code>U(&theta;)</code> (Qiskit's <code>RealAmplitudes</code>) is trained so that, after encoding a data sample
    as a quantum state, a chosen subset of "trash" qubits collapses toward <code>|0\u2026 0&#10217;</code>. Because
    <code>U</code> is unitary, discarding those trash qubits and applying <code>U(&theta;)&#8224;</code> reconstructs
    an approximation of the original state \u2014 the quantum analogue of PCA's encode \u2192 truncate \u2192 decode
    pipeline. Reconstruction quality is measured with a SWAP test against a fresh copy of the true input state.</p>
    <p>Classical PCA (scikit-learn) is run on the identical, reduced-and-normalized feature matrix, with
    <code>n_components</code> chosen to match the quantum method's latent-register capacity, for a like-for-like
    comparison.</p>
  </section>

  <section id="experiments">
    <h2>Experiment matrix</h2>
    <p class="lede">Mirrors the assignment guidelines' recommended E1\u2013E8 structure, using Team 4's own fixed
    parameters (qubits, depth, noise), run on both datasets with repeated trials at 1024 shots.</p>
    {build_matrix_table()}
  </section>

  {"".join(dataset_sections)}

  <section id="limitations">
    <h2>Limitations &amp; proposed improvements</h2>
    <ul class="limits">
      <li><strong>Zero-padding cost.</strong> Wine at 6 qubits is 51/64 zero-padded, so those results mostly measure
      how the autoencoder handles empty amplitude space rather than real feature reduction. Improvement: PCA-informed
      feature selection, or block-encoding two samples per register instead of padding.</li>
      <li><strong>State-preparation overhead.</strong> <code>initialize()</code> costs gates that grow with vector
      length, so reported depth/gate-count partly reflects loading the data, not the autoencoder itself \u2014 a
      known limitation of amplitude-encoded quantum ML. Improvement: report encoder-only depth separately from
      total depth.</li>
      <li><strong>Mini-batch, gradient-free training.</strong> COBYLA optimizes a small fixed subsample and needs a
      large iteration budget to converge well. Improvement: a gradient-based optimizer (parameter-shift + Adam, or
      SPSA) typically reaches a better optimum.</li>
      <li><strong>Evaluation overhead.</strong> The SWAP test needs 2n+1 qubits, more than the encoder's own n
      qubits \u2014 a standard measurement-subroutine cost, reported separately from the encoder's qubit count.</li>
      <li><strong>Simplified noise model.</strong> Depolarizing noise approximates but doesn't capture coherent
      errors, crosstalk or non-Markovian effects. Improvement: import a real backend noise model via
      <code>NoiseModel.from_backend(...)</code> where IBM Quantum access is available.</li>
    </ul>
  </section>

  <footer class="page-foot">
    Prepared by Team 4 for the Quantum Computing course assignment, built with Qiskit, Aer and scikit-learn.
  </footer>
</main>
</body>
</html>"""


CSS = """
:root {
  --bg: #EEF1F6;
  --surface: #FFFFFF;
  --sidebar-bg: #12172B;
  --sidebar-ink: #C7CCE2;
  --ink: #12172B;
  --ink-muted: #5B6478;
  --accent: #4A5FE0;
  --accent-soft: #E7EAFB;
  --classical: #2F8F7E;
  --classical-soft: #E1F1EC;
  --warn: #B4432E;
  --warn-soft: #F7E7E3;
  --border: #DCE1EC;
  --serif: 'IBM Plex Serif', Georgia, 'Times New Roman', serif;
  --sans: 'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
  --mono: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.55;
  display: flex;
  min-height: 100vh;
}
h1, h2 { font-family: var(--serif); font-weight: 600; letter-spacing: -0.01em; }
code {
  font-family: var(--mono);
  background: var(--accent-soft);
  padding: 0.12em 0.4em;
  border-radius: 3px;
  font-size: 0.92em;
}

/* Sidebar */
.sidebar {
  width: 232px;
  flex: 0 0 232px;
  background: var(--sidebar-bg);
  color: var(--sidebar-ink);
  position: sticky;
  top: 0;
  height: 100vh;
  padding: 28px 22px;
  display: flex;
  flex-direction: column;
}
.brand-kicker { font-size: 0.8rem; color: #7C86B8; margin-bottom: 2px; }
.brand-title { font-family: var(--serif); font-size: 1.4rem; color: #fff; font-weight: 700; }
.nav-links { margin-top: 36px; display: flex; flex-direction: column; gap: 4px; flex: 1; }
.nav-links a {
  color: var(--sidebar-ink);
  text-decoration: none;
  font-size: 0.92rem;
  padding: 8px 10px;
  border-radius: 5px;
  border-left: 2px solid transparent;
}
.nav-links a:hover { background: rgba(255,255,255,0.06); border-left-color: var(--accent); color: #fff; }
.sidebar-foot { font-size: 0.76rem; color: #565F84; }

/* Main */
main { flex: 1; padding: 48px 56px 80px; max-width: 980px; }
.page-head { margin-bottom: 48px; }
.page-head h1 { font-size: 2.15rem; margin: 0 0 10px; }
.subtitle { color: var(--ink-muted); max-width: 70ch; font-size: 1.02rem; }

section { margin-bottom: 52px; scroll-margin-top: 20px; }
section h2 { font-size: 1.3rem; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 18px; }
.lede { color: var(--ink-muted); margin-top: -6px; margin-bottom: 18px; max-width: 70ch; }

/* Stat blocks */
.stat-row { display: flex; gap: 20px; flex-wrap: wrap; }
.stat {
  flex: 1 1 160px;
  border-left: 3px solid var(--accent);
  background: var(--surface);
  padding: 14px 16px;
}
.stat-classical { border-left-color: var(--classical); }
.stat-warn { border-left-color: var(--warn); }
.stat-value { font-family: var(--serif); font-size: 1.7rem; font-weight: 700; color: var(--accent); }
.stat-classical .stat-value { color: var(--classical); }
.stat-warn .stat-value { color: var(--warn); }
.stat-label { color: var(--ink-muted); font-size: 0.82rem; margin-top: 4px; }

/* Tables */
table.data-table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  font-size: 0.88rem;
  margin-bottom: 8px;
}
table.data-table th, table.data-table td {
  border-bottom: 1px solid var(--border);
  padding: 8px 10px;
  text-align: left;
}
table.data-table th { color: var(--ink-muted); font-weight: 500; font-size: 0.8rem; }
table.data-table td.num, table.data-table th.num { font-family: var(--mono); text-align: right; }
table.data-table.small { max-width: 480px; }

/* Figures */
.fig-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px; }
.fig-card { background: var(--surface); border: 1px solid var(--border); padding: 12px; margin: 0; }
.fig-card img { width: 100%; display: block; }
.fig-card figcaption { color: var(--ink-muted); font-size: 0.82rem; margin-top: 8px; }

/* Limitations */
ul.limits { padding-left: 0; list-style: none; display: flex; flex-direction: column; gap: 14px; }
ul.limits li { background: var(--surface); border-left: 3px solid var(--border); padding: 12px 16px; }

.page-foot { color: var(--ink-muted); font-size: 0.82rem; border-top: 1px solid var(--border); padding-top: 18px; }

@media print {
  .sidebar { display: none; }
  main { max-width: none; padding: 0 24px; }
  .fig-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 860px) {
  body { flex-direction: column; }
  .sidebar { position: relative; height: auto; width: 100%; flex: none; }
  main { padding: 32px 24px; }
  .fig-grid { grid-template-columns: 1fr; }
}
"""


def main():
    df, agg = load_and_aggregate()
    html = build_html(df, agg)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {OUT_PATH}")
    print("Open it by double-clicking the file, or drag it into a browser window.")


if __name__ == "__main__":
    main()
