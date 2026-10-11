"""
data_utils.py
=============
Team 4 — Quantum Principal Component Analysis
Loading + preprocessing for the "Wine / WDBC reduced" datasets named in the
problem statement.

Why this preprocessing?
------------------------
Amplitude encoding a classical vector x onto n qubits requires:
  1. len(x) == 2**n                      (state-vector dimension)
  2. ||x||_2 == 1                        (valid quantum state)

Neither Wine (13 features) nor WDBC (30 features) is a power of two, so for
every qubit budget n in {2, 4, 6} we "reduce" each dataset to length 2**n by:
  - selecting the top-(2**n) features ranked by variance, if the dataset has
    enough features, else
  - keeping all original features and zero-padding up to 2**n.
Then every row is standardized (zero mean / unit variance per feature, fit on
the TRAIN split only) and finally L2-normalized to unit norm.

The classical PCA baseline is run on the *same* reduced-and-normalized
matrix, so the quantum autoencoder and the classical baseline are always
compared in an identical input space.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE_DEFAULT = 42


@dataclass
class ReducedDataset:
    name: str
    n_qubits: int
    dim: int                 # = 2 ** n_qubits
    n_real_features: int     # how many of `dim` columns are real (non-padded)
    X_train: np.ndarray      # unit-norm rows, shape (n_train, dim)
    X_test: np.ndarray       # unit-norm rows, shape (n_test, dim)
    y_train: np.ndarray
    y_test: np.ndarray


def load_raw(dataset_name: str):
    """dataset_name in {'wine', 'wdbc'} -> (X, y)."""
    dataset_name = dataset_name.lower()
    if dataset_name == "wine":
        d = load_wine()
    elif dataset_name in ("wdbc", "breast_cancer", "cancer"):
        d = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Use 'wine' or 'wdbc'.")
    return d.data.astype(np.float64), d.target.astype(int)


def _select_top_variance_features(X: np.ndarray, k: int) -> np.ndarray:
    """Return column indices of the k highest-variance features."""
    variances = X.var(axis=0)
    order = np.argsort(variances)[::-1]
    return np.sort(order[:k])


def reduce_to_dim(X: np.ndarray, dim: int):
    """
    Reduce X's columns to exactly `dim` columns.
    - If X already has >= dim columns: keep the `dim` highest-variance ones.
    - If X has < dim columns: keep all, zero-pad the rest.
    Returns (X_reduced, n_real_features_kept).
    """
    n_features = X.shape[1]
    if n_features >= dim:
        cols = _select_top_variance_features(X, dim)
        return X[:, cols], dim
    else:
        pad_width = dim - n_features
        X_padded = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
        return X_padded, n_features


def normalize_rows_unit_norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize every row to unit norm (required for amplitude encoding)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    return X / norms


def prepare_dataset(
    dataset_name: str,
    n_qubits: int,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> ReducedDataset:
    """
    Full pipeline: load -> stratified 80:20 train/test split -> choose the
    feature columns FROM THE TRAIN SPLIT ONLY (top-variance features when the
    dataset has >= 2**n features, otherwise keep all) -> apply the SAME columns
    to train and test -> standardize (fit on train only) -> zero-pad to
    2**n_qubits if needed -> unit-normalize.

    BUG FIX: the earlier version chose the columns separately for the train
    and test splits (on standardized data, where every train variance is ~1),
    so train and test could end up using different feature columns.
    """
    dim = 2 ** n_qubits
    X, y = load_raw(dataset_name)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    n_features = X_train_raw.shape[1]
    if n_features >= dim:
        cols = _select_top_variance_features(X_train_raw, dim)  # train split only
        X_train_sel, X_test_sel = X_train_raw[:, cols], X_test_raw[:, cols]
        n_real = dim
    else:
        X_train_sel, X_test_sel = X_train_raw, X_test_raw
        n_real = n_features

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_sel)
    X_test_std = scaler.transform(X_test_sel)

    if n_real < dim:  # zero-pad up to 2**n columns
        pad = dim - n_real
        X_train_std = np.pad(X_train_std, ((0, 0), (0, pad)), mode="constant")
        X_test_std = np.pad(X_test_std, ((0, 0), (0, pad)), mode="constant")

    X_train_norm = normalize_rows_unit_norm(X_train_std)
    X_test_norm = normalize_rows_unit_norm(X_test_std)

    return ReducedDataset(
        name=dataset_name,
        n_qubits=n_qubits,
        dim=dim,
        n_real_features=n_real,
        X_train=X_train_norm,
        X_test=X_test_norm,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    # Quick self-check (run with: python -m src.data_utils)
    for name in ["wine", "wdbc"]:
        for n in [2, 4, 6]:
            ds = prepare_dataset(name, n)
            norms = np.linalg.norm(ds.X_train, axis=1)
            print(
                f"{name:5s} n_qubits={n} dim={ds.dim:3d} "
                f"real_features={ds.n_real_features:2d}/{ds.dim:3d}  "
                f"train={ds.X_train.shape} test={ds.X_test.shape}  "
                f"row-norm mean={norms.mean():.6f} (should be 1.0)"
            )
