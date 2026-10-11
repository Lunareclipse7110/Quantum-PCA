"""
quantum_autoencoder.py
=======================
Team 4 — Quantum Principal Component Analysis
"Quantum-inspired PCA" implemented as a trainable Quantum Autoencoder (QAE),
following Romero, Olson & Aspuru-Guzik (2017), "Quantum autoencoders for
efficient compression of quantum data".

Why a quantum autoencoder for "quantum PCA"?
---------------------------------------------
Classical PCA finds a low-dimensional subspace that best reconstructs the
data. The quantum autoencoder does the direct quantum analogue: it learns a
parameterized unitary U(theta) acting on n_qubits such that, after applying
U(theta) to an amplitude-encoded data sample, a chosen subset of "trash"
qubits collapses (as closely as possible) to |0...0>. Because U is unitary,
throwing away information-free trash qubits and re-applying U(theta)^dagger
reconstructs the original state -- exactly analogous to PCA's
encode -> truncate -> decode pipeline, but performed with a quantum circuit.

Circuit roles
-------------
- ENCODER  : RealAmplitudes(n_qubits, reps=depth)  -- the trainable ansatz.
- TRAINING circuit: |x> --initialize--> encoder(theta) --> measure trash
  qubits. Cost = 1 - P(trash qubits all read 0), estimated over `shots`.
- EVALUATION circuit (reconstruction fidelity): a SWAP test between
  (a) register A = initialize(x) -> encoder(theta) -> reset trash qubits
      -> decoder(theta) [= encoder(theta)^dagger], and
  (b) register B = a fresh initialize(x) reference copy.
  Fidelity = 2*P(ancilla=0) - 1, reconstruction_error = 1 - fidelity.

All circuits are transpiled to the fixed basis gate set in noise_models.py
before being run or before resource metrics (qubits/depth/gate count) are
recorded, and noise is injected only through the AerSimulator's noise_model.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .noise_models import build_noise_model, circuit_resource_report, transpile_for_backend


# --------------------------------------------------------------------------- #
# Circuit construction
# --------------------------------------------------------------------------- #
def build_encoder_ansatz(n_qubits: int, depth: int) -> RealAmplitudes:
    """The trainable encoder U(theta). `depth` = number of ansatz repetitions
    (an *input* experimental parameter, distinct from the *measured*
    transpiled circuit.depth() reported alongside every result)."""
    return RealAmplitudes(num_qubits=n_qubits, reps=depth, entanglement="linear")


def trash_qubit_indices(n_qubits: int, trash_qubits: int) -> List[int]:
    """Fixed convention: the LAST `trash_qubits` qubits are the trash
    register; the remaining leading qubits are the latent register."""
    return list(range(n_qubits - trash_qubits, n_qubits))


def build_training_circuit(
    n_qubits: int, trash_qubits: int, encoder_bound: QuantumCircuit, x: np.ndarray
) -> QuantumCircuit:
    """initialize(x) -> encoder(theta) -> measure trash qubits."""
    qc = QuantumCircuit(n_qubits, trash_qubits)
    qc.initialize(x, list(range(n_qubits)))
    qc.compose(encoder_bound, qubits=list(range(n_qubits)), inplace=True)
    qc.measure(trash_qubit_indices(n_qubits, trash_qubits), list(range(trash_qubits)))
    return qc


def build_swap_test_circuit(
    n_qubits: int, trash_qubits: int, encoder_bound: QuantumCircuit, x: np.ndarray
) -> QuantumCircuit:
    """
    SWAP-test circuit estimating the fidelity between the reconstructed
    state (register A) and the true input state (register B, untouched).
    """
    qA = QuantumRegister(n_qubits, "A")
    qB = QuantumRegister(n_qubits, "B")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qA, qB, anc, creg)

    # --- Register A: encode -> reset trash -> decode ---
    qc.initialize(x, list(qA))
    qc.compose(encoder_bound, qubits=list(qA), inplace=True)
    for idx in trash_qubit_indices(n_qubits, trash_qubits):
        qc.reset(qA[idx])
    decoder_bound = encoder_bound.inverse()
    qc.compose(decoder_bound, qubits=list(qA), inplace=True)

    # --- Register B: untouched reference copy ---
    qc.initialize(x, list(qB))

    # --- SWAP test ---
    qc.h(anc[0])
    for i in range(n_qubits):
        qc.cswap(anc[0], qA[i], qB[i])
    qc.h(anc[0])
    qc.measure(anc[0], creg[0])
    return qc


def _prob_all_zero_trash(counts: dict, trash_qubits: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    zero_key = "0" * trash_qubits
    return counts.get(zero_key, 0) / total


def _prob_ancilla_zero(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    n_zero = sum(c for bitstring, c in counts.items() if bitstring[-1] == "0")
    return n_zero / total


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    theta: np.ndarray
    cost_history: List[float] = field(default_factory=list)
    n_iterations: int = 0
    train_time_s: float = 0.0


def _training_cost_batch(theta, ansatz, backend, X_batch, trash_qubits, shots) -> float:
    n_qubits = ansatz.num_qubits
    encoder_bound = ansatz.assign_parameters(theta)
    circuits = [
        build_training_circuit(n_qubits, trash_qubits, encoder_bound, x) for x in X_batch
    ]
    tcircuits = [transpile_for_backend(qc, backend) for qc in circuits]
    result = backend.run(tcircuits, shots=shots).result()
    p_zeros = [
        _prob_all_zero_trash(result.get_counts(i), trash_qubits) for i in range(len(tcircuits))
    ]
    return 1.0 - float(np.mean(p_zeros))


def train_autoencoder(
    ansatz: RealAmplitudes,
    backend: AerSimulator,
    X_train_subset: np.ndarray,
    trash_qubits: int,
    shots: int = 1024,
    seed: int = 0,
    maxiter: int = 150,
) -> TrainingResult:
    """Trains theta with COBYLA against a FIXED mini-training-set (drawn once
    by the caller) so the objective is deterministic between iterations --
    important for COBYLA's trust-region search. See README 'Limitations' for
    the trade-off this implies (mini-batch, not full-dataset, training)."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    history: List[float] = []

    def objective(theta):
        cost = _training_cost_batch(theta, ansatz, backend, X_train_subset, trash_qubits, shots)
        history.append(cost)
        return cost

    t0 = time.perf_counter()
    result = minimize(
        objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 1.0}
    )
    train_time = time.perf_counter() - t0

    return TrainingResult(
        theta=result.x, cost_history=history, n_iterations=len(history), train_time_s=train_time
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    mean_fidelity: float
    std_fidelity: float
    mean_reconstruction_error: float
    eval_time_s: float
    example_counts: Optional[dict] = None  # one sample's ancilla counts, for a histogram


def evaluate_reconstruction(
    ansatz: RealAmplitudes,
    theta: np.ndarray,
    backend: AerSimulator,
    X_eval: np.ndarray,
    trash_qubits: int,
    shots: int = 1024,
) -> EvaluationResult:
    n_qubits = ansatz.num_qubits
    encoder_bound = ansatz.assign_parameters(theta)

    fidelities = []
    example_counts = None

    t0 = time.perf_counter()
    circuits = [
        build_swap_test_circuit(n_qubits, trash_qubits, encoder_bound, x) for x in X_eval
    ]
    tcircuits = [transpile_for_backend(qc, backend) for qc in circuits]
    result = backend.run(tcircuits, shots=shots).result()

    for i in range(len(tcircuits)):
        counts = result.get_counts(i)
        if i == 0:
            example_counts = counts
        p0 = _prob_ancilla_zero(counts)
        fidelity = max(0.0, 2.0 * p0 - 1.0)
        fidelities.append(fidelity)
    eval_time = time.perf_counter() - t0

    fidelities = np.array(fidelities)
    return EvaluationResult(
        mean_fidelity=float(fidelities.mean()),
        std_fidelity=float(fidelities.std()),
        mean_reconstruction_error=float(1.0 - fidelities.mean()),
        eval_time_s=eval_time,
        example_counts=example_counts,
    )


# --------------------------------------------------------------------------- #
# One full experiment configuration (train + evaluate + resource report)
# --------------------------------------------------------------------------- #
@dataclass
class QAEExperimentResult:
    n_qubits: int
    depth: int
    trash_qubits: int
    latent_qubits: int
    noise_level: str
    resource_report: dict
    training: TrainingResult
    evaluation: EvaluationResult


def run_qae_experiment(
    n_qubits: int,
    depth: int,
    noise_level: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    shots: int = 1024,
    n_train_samples: int = 12,
    n_eval_samples: int = 20,
    seed: int = 0,
    maxiter: int = 150,
) -> QAEExperimentResult:
    trash_qubits = max(1, n_qubits // 2)
    latent_qubits = n_qubits - trash_qubits

    noise_model = build_noise_model(noise_level)
    # seed_simulator makes the shot sampling reproducible (same seed -> same counts).
    if noise_model is not None:
        backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    else:
        backend = AerSimulator(seed_simulator=seed)

    ansatz = build_encoder_ansatz(n_qubits, depth)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(X_train), size=min(n_train_samples, len(X_train)), replace=False)
    eval_idx = rng.choice(len(X_test), size=min(n_eval_samples, len(X_test)), replace=False)
    X_train_subset = X_train[train_idx]
    X_eval_subset = X_test[eval_idx]

    training = train_autoencoder(
        ansatz, backend, X_train_subset, trash_qubits, shots=shots, seed=seed, maxiter=maxiter
    )
    evaluation = evaluate_reconstruction(
        ansatz, training.theta, backend, X_eval_subset, trash_qubits, shots=shots
    )

    # Resource report from a representative bound training circuit.
    encoder_bound = ansatz.assign_parameters(training.theta)
    sample_circuit = build_training_circuit(n_qubits, trash_qubits, encoder_bound, X_train[0])
    transpiled = transpile_for_backend(sample_circuit, backend)
    resource_report = circuit_resource_report(transpiled)

    return QAEExperimentResult(
        n_qubits=n_qubits,
        depth=depth,
        trash_qubits=trash_qubits,
        latent_qubits=latent_qubits,
        noise_level=noise_level,
        resource_report=resource_report,
        training=training,
        evaluation=evaluation,
    )
