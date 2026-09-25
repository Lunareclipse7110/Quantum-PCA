"""
noise_models.py
================
Team 4 — Quantum Principal Component Analysis
Builds the three noise conditions fixed by the assignment guidelines
(Section B: "Noise conditions: N0 no noise / N1 low noise / N2 medium
noise"), relabelled to match Team 4's own row: "Ideal, low, moderate".

All circuits are transpiled to the fixed basis gate set BASIS_GATES before
noise is attached, so the noise model only ever needs to know about 1- and
2-qubit basis gates plus measurement/readout error. This avoids any
mismatch between "logical" gates (RY, CX, CSWAP, ...) used when building
circuits and the "physical" gates the simulator actually executes.
"""
from qiskit import transpile
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# Standard IBM-style basis: every circuit is transpiled to this gate set
# before noise is applied and before depth/gate-count metrics are recorded.
BASIS_GATES = ["id", "rz", "sx", "x", "cx"]

# (p1 = single-qubit depolarizing prob, p2 = two-qubit depolarizing prob,
#  p_ro = symmetric readout bit-flip prob)
NOISE_LEVELS = {
    "ideal": None,
    "low": dict(p1=0.001, p2=0.010, p_ro=0.01),
    "moderate": dict(p1=0.005, p2=0.030, p_ro=0.05),
}


def build_noise_model(level: str):
    """level in {'ideal', 'low', 'moderate'} -> NoiseModel or None."""
    if level not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level '{level}'. Use one of {list(NOISE_LEVELS)}")

    params = NOISE_LEVELS[level]
    if params is None:  # ideal
        return None

    nm = NoiseModel(basis_gates=BASIS_GATES)

    error_1q = depolarizing_error(params["p1"], 1)
    error_2q = depolarizing_error(params["p2"], 2)

    nm.add_all_qubit_quantum_error(error_1q, ["id", "rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(error_2q, ["cx"])

    p_ro = params["p_ro"]
    readout_error = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_readout_error(readout_error)

    return nm


def transpile_for_backend(circuit, backend=None, optimization_level: int = 1):
    """Transpile to the fixed basis gate set used for noise + resource reporting.

    `backend` is accepted for readability at call sites but intentionally
    NOT passed to `transpile()`: these circuits are small and fully
    connected, so an explicit `basis_gates` list is sufficient and avoids
    any backend-specific coupling-map/layout behaviour.
    """
    return transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=optimization_level,
    )


def circuit_resource_report(transpiled_circuit) -> dict:
    """Number of qubits, circuit depth, and gate counts for the deliverables in
    Section A of the guidelines ('Number of qubits', 'Number of gates',
    'Circuit depth')."""
    ops = transpiled_circuit.count_ops()
    return {
        "n_qubits": transpiled_circuit.num_qubits,
        "depth": transpiled_circuit.depth(),
        "n_gates_total": sum(ops.values()),
        "gate_counts": dict(ops),
    }
