"""
visualize.py
=============
Team 4 — Quantum Principal Component Analysis
Builds every plot referenced in the guidelines' "Histograms/graphs"
deliverable, from results/results.csv (+ the saved training curves and
example measurement counts).

Usage
-----
    python -m src.visualize
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
COUNTS_DIR = os.path.join(RESULTS_DIR, "example_counts")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def _load():
    df = pd.read_csv(RESULTS_CSV)
    return df


def _agg(df):
    """Mean +/- std across repeats, grouped by (dataset, exp_id)."""
    group_cols = ["dataset", "exp_id", "n_qubits", "depth", "noise_level"]
    metrics = [
        "q_reconstruction_error", "q_mean_fidelity", "q_circuit_depth", "q_n_gates_total",
        "q_n_two_qubit_gates",
        "q_train_time_s", "q_eval_time_s",
        "c_reconstruction_mse", "c_explained_variance_ratio_sum", "c_fit_time_s",
    ]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.values]
    return agg


def plot_reconstruction_quality(agg: pd.DataFrame, dataset: str, out_path: str):
    sub = agg[agg["dataset"] == dataset].sort_values("exp_id")
    x = np.arange(len(sub))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, sub["q_reconstruction_error_mean"], width,
           yerr=sub["q_reconstruction_error_std"].fillna(0), label="Quantum autoencoder", capsize=3)
    ax.bar(x + width / 2, sub["c_reconstruction_mse_mean"], width,
           yerr=sub["c_reconstruction_mse_std"].fillna(0), label="Classical PCA", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["exp_id"])
    ax.set_xlabel("Experiment (see experiment_matrix.py)")
    ax.set_ylabel("Reconstruction error")
    ax.set_title(f"Reconstruction quality: Quantum PCA vs Classical PCA ({dataset.upper()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_noise_sensitivity(agg: pd.DataFrame, dataset: str, out_path: str):
    """Uses E4 (ideal), E5 (low), E6 (moderate): n_qubits=4, depth=4 fixed."""
    sub = agg[(agg["dataset"] == dataset) & (agg["exp_id"].isin(["E4", "E5", "E6"]))]
    order = {"ideal": 0, "low": 1, "moderate": 2}
    sub = sub.assign(_order=sub["noise_level"].map(order)).sort_values("_order")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        sub["noise_level"], sub["q_reconstruction_error_mean"],
        yerr=sub["q_reconstruction_error_std"].fillna(0),
        marker="o", capsize=4, label="Quantum autoencoder",
    )
    ax.axhline(
        sub["c_reconstruction_mse_mean"].iloc[0], linestyle="--", color="gray",
        label="Classical PCA (noise-free baseline)",
    )
    ax.set_xlabel("Noise level (n_qubits=4, depth=4)")
    ax.set_ylabel("Reconstruction error")
    ax.set_title(f"Noise sensitivity of reconstruction quality ({dataset.upper()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_resource_usage(agg: pd.DataFrame, dataset: str, out_path: str):
    sub = agg[agg["dataset"] == dataset].sort_values("exp_id")
    x = np.arange(len(sub))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(x, sub["q_circuit_depth_mean"], color="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["exp_id"])
    ax1.set_ylabel("Transpiled circuit depth")
    ax1.set_title("Circuit depth per experiment")

    ax2.bar(x, sub["q_n_gates_total_mean"], color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["exp_id"])
    ax2.set_ylabel("Total gate count")
    ax2.set_title("Gate count per experiment")

    fig.suptitle(f"Circuit resources ({dataset.upper()})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve(dataset: str, exp_id: str, out_path: str):
    path = os.path.join(CURVES_DIR, f"{dataset}_{exp_id}.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        history = json.load(f)
    if not history:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(history) + 1), history, marker=".")
    ax.set_xlabel("COBYLA iteration")
    ax.set_ylabel("Training cost  (1 - P(trash qubits = 0))")
    ax.set_title(f"Training convergence: {dataset.upper()} {exp_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_measurement_histogram(dataset: str, exp_ids, labels, out_path: str):
    """Side-by-side ancilla-measurement histograms, e.g. ideal vs moderate noise."""
    fig, axes = plt.subplots(1, len(exp_ids), figsize=(5 * len(exp_ids), 4.5), squeeze=False)
    axes = axes[0]
    for ax, exp_id, label in zip(axes, exp_ids, labels):
        path = os.path.join(COUNTS_DIR, f"{dataset}_{exp_id}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            counts = json.load(f)
        keys = sorted(counts.keys())
        values = [counts[k] for k in keys]
        ax.bar(keys, values, color="tab:green")
        ax.set_xlabel("Ancilla measurement outcome")
        ax.set_ylabel("Counts (shots)")
        ax.set_title(f"{label}\n({dataset.upper()} {exp_id})")
    fig.suptitle("SWAP-test ancilla measurement histogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_all_plots():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = _load()
    agg = _agg(df)

    for dataset in df["dataset"].unique():
        plot_reconstruction_quality(agg, dataset, os.path.join(FIG_DIR, f"{dataset}_reconstruction_quality.png"))
        plot_noise_sensitivity(agg, dataset, os.path.join(FIG_DIR, f"{dataset}_noise_sensitivity.png"))
        plot_resource_usage(agg, dataset, os.path.join(FIG_DIR, f"{dataset}_resource_usage.png"))
        plot_training_curve(dataset, "E4", os.path.join(FIG_DIR, f"{dataset}_training_curve_E4.png"))
        plot_measurement_histogram(
            dataset, ["E4", "E6"], ["Ideal noise", "Moderate noise"],
            os.path.join(FIG_DIR, f"{dataset}_measurement_histogram_E4_vs_E6.png"),
        )

    agg.to_csv(os.path.join(RESULTS_DIR, "results_aggregated.csv"), index=False)
    print(f"Figures written to {FIG_DIR}")
    print(f"Aggregated table written to {os.path.join(RESULTS_DIR, 'results_aggregated.csv')}")


if __name__ == "__main__":
    make_all_plots()
