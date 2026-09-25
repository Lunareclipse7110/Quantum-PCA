"""
run_experiments.py
====================
Team 4 — Quantum Principal Component Analysis
Runs EXPERIMENT_MATRIX x DATASETS x repeated trials, comparing the quantum
autoencoder ("quantum PCA") against classical PCA on every configuration,
and writes results/results.csv.

Usage
-----
    python -m src.run_experiments                     # full run (all 8 x 2 x 3)
    python -m src.run_experiments --repeats 1 --quick  # fast smoke test
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from .classical_baseline import n_components_for_latent_qubits, run_classical_pca
from .data_utils import prepare_dataset
from .experiment_matrix import DATASETS, EXPERIMENT_MATRIX
from .quantum_autoencoder import run_qae_experiment

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
COUNTS_DIR = os.path.join(RESULTS_DIR, "example_counts")

FIELDNAMES = [
    "exp_id", "dataset", "repeat", "n_qubits", "depth", "noise_level",
    "trash_qubits", "latent_qubits",
    "q_final_training_cost", "q_train_iterations", "q_train_time_s",
    "q_mean_fidelity", "q_std_fidelity", "q_reconstruction_error",
    "q_eval_time_s", "q_circuit_depth", "q_n_gates_total",
    "c_n_components", "c_explained_variance_ratio_sum", "c_reconstruction_mse",
    "c_fit_time_s", "c_transform_time_s",
]


def run_all(repeats: int = 3, n_train_samples: int = 16, n_eval_samples: int = 20,
            maxiter: int = 150, shots: int = 1024, verbose: bool = True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)
    os.makedirs(COUNTS_DIR, exist_ok=True)
    rows = []

    for dataset_name in DATASETS:
        for cfg in EXPERIMENT_MATRIX:
            ds = prepare_dataset(dataset_name, cfg.n_qubits)
            latent_qubits = cfg.n_qubits - max(1, cfg.n_qubits // 2)
            n_components = n_components_for_latent_qubits(latent_qubits)
            classical = run_classical_pca(ds, n_components)

            for repeat in range(repeats):
                seed = 1000 * repeat + hash((dataset_name, cfg.exp_id)) % 1000
                t0 = time.perf_counter()
                qae_result = run_qae_experiment(
                    n_qubits=cfg.n_qubits,
                    depth=cfg.depth,
                    noise_level=cfg.noise_level,
                    X_train=ds.X_train,
                    X_test=ds.X_test,
                    shots=shots,
                    n_train_samples=n_train_samples,
                    n_eval_samples=n_eval_samples,
                    seed=seed,
                    maxiter=maxiter,
                )
                wall_time = time.perf_counter() - t0

                row = {
                    "exp_id": cfg.exp_id,
                    "dataset": dataset_name,
                    "repeat": repeat,
                    "n_qubits": cfg.n_qubits,
                    "depth": cfg.depth,
                    "noise_level": cfg.noise_level,
                    "trash_qubits": qae_result.trash_qubits,
                    "latent_qubits": qae_result.latent_qubits,
                    "q_final_training_cost": qae_result.training.cost_history[-1]
                        if qae_result.training.cost_history else float("nan"),
                    "q_train_iterations": qae_result.training.n_iterations,
                    "q_train_time_s": qae_result.training.train_time_s,
                    "q_mean_fidelity": qae_result.evaluation.mean_fidelity,
                    "q_std_fidelity": qae_result.evaluation.std_fidelity,
                    "q_reconstruction_error": qae_result.evaluation.mean_reconstruction_error,
                    "q_eval_time_s": qae_result.evaluation.eval_time_s,
                    "q_circuit_depth": qae_result.resource_report["depth"],
                    "q_n_gates_total": qae_result.resource_report["n_gates_total"],
                    "c_n_components": classical.n_components,
                    "c_explained_variance_ratio_sum": classical.explained_variance_ratio_sum,
                    "c_reconstruction_mse": classical.reconstruction_mse,
                    "c_fit_time_s": classical.fit_time_s,
                    "c_transform_time_s": classical.transform_time_s,
                }
                rows.append(row)

                # Save one representative training curve + measurement
                # histogram per (dataset, exp_id) -- from the first repeat --
                # for the "histograms/graphs" and "measurement results"
                # deliverables in Section A of the guidelines.
                if repeat == 0:
                    curve_path = os.path.join(CURVES_DIR, f"{dataset_name}_{cfg.exp_id}.json")
                    with open(curve_path, "w") as f:
                        json.dump(qae_result.training.cost_history, f)

                    counts_path = os.path.join(COUNTS_DIR, f"{dataset_name}_{cfg.exp_id}.json")
                    with open(counts_path, "w") as f:
                        json.dump(qae_result.evaluation.example_counts or {}, f)

                if verbose:
                    print(
                        f"[{dataset_name:5s} {cfg.exp_id}] n={cfg.n_qubits} d={cfg.depth} "
                        f"noise={cfg.noise_level:8s} rep={repeat} -> "
                        f"Q recon_err={row['q_reconstruction_error']:.4f}  "
                        f"C recon_mse={row['c_reconstruction_mse']:.4f}  "
                        f"(wall {wall_time:.1f}s)"
                    )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"\nSaved {len(rows)} rows -> {RESULTS_CSV}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Team 4 Quantum PCA experiment matrix.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated runs per configuration.")
    parser.add_argument("--quick", action="store_true", help="Fast smoke test (tiny settings).")
    args = parser.parse_args()

    if args.quick:
        run_all(repeats=1, n_train_samples=4, n_eval_samples=6, maxiter=8)
    else:
        run_all(repeats=args.repeats)
