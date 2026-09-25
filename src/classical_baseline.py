"""
classical_baseline.py
======================
Team 4 — Quantum Principal Component Analysis
Classical PCA baseline required by the guidelines ("Classical baseline:
Classical PCA"). Run on the *same* reduced + unit-normalized feature matrix
the quantum autoencoder uses, so both methods are compared in an identical
input space.
"""
import time
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .data_utils import ReducedDataset


@dataclass
class ClassicalPCAResult:
    n_components: int
    explained_variance_ratio_sum: float
    reconstruction_mse: float
    fit_time_s: float
    transform_time_s: float


def n_components_for_latent_qubits(latent_qubits: int) -> int:
    """
    Give classical PCA the same *representational capacity* as the quantum
    autoencoder's latent register: `latent_qubits` qubits can represent a
    2**latent_qubits - dimensional subspace, so we let classical PCA keep
    that many principal components. This is the basis for an apples-to-
    apples "reduction quality" comparison (see README for the reasoning).
    """
    return 2 ** latent_qubits


def run_classical_pca(ds: ReducedDataset, n_components: int) -> ClassicalPCAResult:
    n_components = min(n_components, ds.X_train.shape[1], ds.X_train.shape[0])

    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(ds.X_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test_proj = pca.transform(ds.X_test)
    X_test_recon = pca.inverse_transform(X_test_proj)
    transform_time = time.perf_counter() - t0

    mse = float(np.mean((ds.X_test - X_test_recon) ** 2))
    ev_ratio_sum = float(np.sum(pca.explained_variance_ratio_))

    return ClassicalPCAResult(
        n_components=n_components,
        explained_variance_ratio_sum=ev_ratio_sum,
        reconstruction_mse=mse,
        fit_time_s=fit_time,
        transform_time_s=transform_time,
    )


if __name__ == "__main__":
    from .data_utils import prepare_dataset

    for name in ["wine", "wdbc"]:
        for n_qubits in [2, 4, 6]:
            ds = prepare_dataset(name, n_qubits)
            latent = n_qubits // 2
            k = n_components_for_latent_qubits(latent)
            res = run_classical_pca(ds, k)
            print(
                f"{name:5s} n_qubits={n_qubits} latent_qubits={latent} "
                f"classical_k={res.n_components:2d}  "
                f"EVR_sum={res.explained_variance_ratio_sum:.4f}  "
                f"recon_MSE={res.reconstruction_mse:.5f}  "
                f"fit_time={res.fit_time_s*1000:.2f}ms"
            )
