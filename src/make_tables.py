"""
make_tables.py
===============
Team 4 — Quantum Principal Component Analysis
Builds the report tables straight from results/results.csv so the numbers in
the report always come from the SAME run as the figures and screenshots.

Writes (in results/tables/):
    table_<dataset>.csv / .md   -> "Results table" (Table 2 / Table 3 format):
        primary metric (recon. error, mean ± SD), secondary metric
        (fidelity, mean ± SD), total gates, two-qubit gates, depth,
        classical-PCA baseline MSE
    extra_<dataset>.csv / .md   -> explained variance + runtime table
        (classical PCA explained variance, PCA fit time, QAE train / eval time)

Usage:  python -m src.make_tables
"""
import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def _pm(series, nd=4):
    """'mean ± SD' (sample SD, ddof=1; 0 if only one repeat)."""
    sd = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{series.mean():.{nd}f} ± {sd:.{nd}f}"


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def make_tables():
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    n_rep = int(df["repeat"].nunique())

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        main_rows, extra_rows = [], []
        for exp_id, g in sub.groupby("exp_id", sort=True):
            main_rows.append({
                "Exp": exp_id,
                "Primary: recon. error (mean ± SD)": _pm(g["q_reconstruction_error"]),
                "Secondary: fidelity (mean ± SD)": _pm(g["q_mean_fidelity"]),
                "Total gates": int(round(g["q_n_gates_total"].mean())),
                "Two-qubit gates": int(round(g["q_n_two_qubit_gates"].mean())),
                "Depth": int(round(g["q_circuit_depth"].mean())),
                "Baseline (PCA MSE)": f"{g['c_reconstruction_mse'].mean():.4f}",
            })
            extra_rows.append({
                "Exp": exp_id,
                "PCA components": int(g["c_n_components"].iloc[0]),
                "PCA explained variance (sum)": f"{g['c_explained_variance_ratio_sum'].mean():.4f}",
                "PCA fit time (ms)": f"{g['c_fit_time_s'].mean() * 1000:.2f}",
                "QAE train time (s)": _pm(g["q_train_time_s"], 1),
                "QAE eval time (s)": _pm(g["q_eval_time_s"], 1),
            })
        main_df, extra_df = pd.DataFrame(main_rows), pd.DataFrame(extra_rows)

        main_df.to_csv(os.path.join(TABLE_DIR, f"table_{dataset}.csv"), index=False)
        extra_df.to_csv(os.path.join(TABLE_DIR, f"extra_{dataset}.csv"), index=False)
        with open(os.path.join(TABLE_DIR, f"table_{dataset}.md"), "w", encoding="utf-8") as f:
            f.write(f"Results for {dataset.upper()} ({n_rep} repeats; SD = sample SD)\n\n")
            f.write(_to_markdown(main_df) + "\n")
        with open(os.path.join(TABLE_DIR, f"extra_{dataset}.md"), "w", encoding="utf-8") as f:
            f.write(f"Explained variance and runtime, {dataset.upper()}\n\n")
            f.write(_to_markdown(extra_df) + "\n")

        print(f"\n=== {dataset.upper()} results table ===")
        print(_to_markdown(main_df))
        print(f"\n=== {dataset.upper()} explained variance / runtime ===")
        print(_to_markdown(extra_df))

    print(f"\nTables written to {TABLE_DIR}")


if __name__ == "__main__":
    make_tables()
