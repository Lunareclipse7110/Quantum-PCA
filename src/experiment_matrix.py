"""
experiment_matrix.py
=====================
Team 4 — Quantum Principal Component Analysis
Mirrors the "Recommended Experimental Matrix" structure in Section D of the
guidelines (E1-E8: vary two parameters at a time, don't test every
combination), built from Team 4's own fixed parameters:

    Qubits : 2, 4, 6
    Depth  : 2, 4
    Noise  : Ideal, Low, Moderate

This satisfies the Section C mandatory requirements: at least 6 controlled
experiments (we use 8), varying at least two parameters (qubits, depth AND
noise all vary here), all at 1024 shots.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    exp_id: str
    n_qubits: int
    depth: int
    noise_level: str  # "ideal" | "low" | "moderate"


EXPERIMENT_MATRIX = [
    ExperimentConfig("E1", n_qubits=2, depth=2, noise_level="ideal"),
    ExperimentConfig("E2", n_qubits=2, depth=4, noise_level="ideal"),
    ExperimentConfig("E3", n_qubits=4, depth=2, noise_level="ideal"),
    ExperimentConfig("E4", n_qubits=4, depth=4, noise_level="ideal"),
    ExperimentConfig("E5", n_qubits=4, depth=4, noise_level="low"),
    ExperimentConfig("E6", n_qubits=4, depth=4, noise_level="moderate"),
    ExperimentConfig("E7", n_qubits=6, depth=4, noise_level="ideal"),
    ExperimentConfig("E8", n_qubits=6, depth=4, noise_level="moderate"),
]

DATASETS = ["wine", "wdbc"]
