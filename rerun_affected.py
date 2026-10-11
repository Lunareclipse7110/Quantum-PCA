"""
rerun_affected.py
==================
Team 4 — Quantum Principal Component Analysis

WHAT THIS DOES
--------------
The original data pipeline had a bug: when a dataset had at least 2**n features
(WDBC at 2 and 4 qubits, Wine at 2 qubits) the feature columns were chosen
separately for the train and test splits, so train and test could use
DIFFERENT columns. Only these 8 (dataset, experiment) groups are affected:

    WDBC : E1, E2, E3, E4, E5, E6
    Wine : E1, E2

Everything else (WDBC E7/E8, Wine E3-E8) only zero-pads, so it is identical
before and after the fix and is KEPT from your existing results.csv.

This script re-runs only the 8 affected groups with the fixed data pipeline and
merges them into results/results.csv, so you get one complete, consistent
table WITHOUT re-running the slow 6-qubit experiments (~40-50 minutes total).

To keep the merged results comparable with the rows you already have, it uses
the SAME settings as your original run: 70:30 split, original noise values,
3 repeats. (The report must state these settings.)

USAGE
-----
    python rerun_affected.py                 # real run, ~40-50 min
    python rerun_affected.py --repeats 3     # same thing (3 is the default)
Then it automatically rebuilds figures, tables and the HTML report.
"""
import argparse
import os
import shutil

import pandas as pd

import src.noise_models as nm

# --- keep the ORIGINAL run's settings so merged rows are comparable ---------
nm.NOISE_LEVELS["low"] = dict(p1=0.001, p2=0.010, p_ro=0.01)
nm.NOISE_LEVELS["moderate"] = dict(p1=0.005, p2=0.030, p_ro=0.05)

import src.data_utils as du  # noqa: E402
import src.run_experiments as rx  # noqa: E402

_prepare = du.prepare_dataset
rx.prepare_dataset = lambda name, n_qubits, **kw: _prepare(name, n_qubits, test_size=0.3, random_state=42)

JOBS = {"wdbc": ["E1", "E2", "E3", "E4", "E5", "E6"], "wine": ["E1", "E2"]}
# CX count = initialize() CX + ansatz CX (reps*(n-1)); matches the transpiled counts.
INIT_CX = {2: 1, 4: 11, 6: 57}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="e.g. wine:E1 (for testing)")
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=20)
    args = ap.parse_args()

    results_csv = rx.RESULTS_CSV
    backup = os.path.join(rx.RESULTS_DIR, "results_before_fix.csv")
    if not os.path.exists(backup):
        shutil.copy(results_csv, backup)
        print(f"Backed up original results -> {backup}")

    jobs = {d: list(e) for d, e in JOBS.items()}
    if args.only:
        jobs = {}
        for item in args.only:
            d, e = item.split(":")
            jobs.setdefault(d, []).append(e)

    full_matrix = list(rx.EXPERIMENT_MATRIX)
    tmp_csv = os.path.join(rx.RESULTS_DIR, "_rerun_tmp.csv")
    new_frames = []
    for dataset, exps in jobs.items():
        rx.DATASETS = [dataset]
        rx.EXPERIMENT_MATRIX = [c for c in full_matrix if c.exp_id in exps]
        rx.RESULTS_CSV = tmp_csv
        rx.run_all(repeats=args.repeats, n_train_samples=args.n_train,
                   n_eval_samples=args.n_eval, maxiter=args.maxiter)
        new_frames.append(pd.read_csv(tmp_csv))
    os.remove(tmp_csv)
    new = pd.concat(new_frames, ignore_index=True)

    old = pd.read_csv(backup)
    if "q_n_two_qubit_gates" not in old.columns:
        old["q_n_two_qubit_gates"] = old["n_qubits"].map(INIT_CX) + old["depth"] * (old["n_qubits"] - 1)
    redone = set(zip(new["dataset"], new["exp_id"]))
    keep = old[[(d, e) not in redone for d, e in zip(old["dataset"], old["exp_id"])]]
    merged = pd.concat([keep, new], ignore_index=True)
    merged["_o"] = merged["exp_id"].str[1:].astype(int)
    merged = merged.sort_values(["dataset", "_o", "repeat"], ascending=[False, True, True]).drop(columns="_o")
    merged = merged[rx.FIELDNAMES]
    merged.to_csv(results_csv, index=False)
    print(f"\nMerged: kept {len(keep)} old rows + {len(new)} new rows = {len(merged)} -> {results_csv}")

    from src.generate_report import main as generate_report
    from src.make_tables import make_tables
    from src.visualize import make_all_plots
    make_all_plots()
    make_tables()
    generate_report()
    print("\nDone. New tables: results/tables/*.md ; figures: results/figures/*.png")


if __name__ == "__main__":
    main()
