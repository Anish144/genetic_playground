"""NumPy kernels. A kernel is a callable ``k(X, X2=None)`` returning the (n, n2) Gram matrix.

Inputs are (n, d) arrays; 1-D inputs are treated as a single column.
"""

import abc

import numpy as np
from scipy.spatial.distance import pdist


def square_distance(X: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (n, n2)."""
    sq = np.sum(X**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X @ X2.T
    return np.maximum(sq, 0.0)  # cancellation can give tiny negatives


def median_heuristic(X: np.ndarray, max_points: int = 2000) -> float:
    """Median pairwise Euclidean distance between rows of X, a default RBF lengthscale.

    Uses an evenly spaced subsample of at most ``max_points`` rows, so pooled data stacked by
    domain is still covered evenly.
    """
    X = _as_2d(X)
    X = X[:: -(-len(X) // max_points)]  # stride = ceil(n / max_points)
    return float(np.median(pdist(X)))


class Kernel(abc.ABC):
    def __call__(self, X: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X = _as_2d(X)
        return self.K(X, X if X2 is None else _as_2d(X2))

    @abc.abstractmethod
    def K(self, X: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Gram matrix between the rows of X (n, d) and X2 (n2, d), shape (n, n2)."""


class RBF(Kernel):
    """k(x, x') = exp(-‖x - x'‖² / (2ℓ²)). An array of lengthscales gives ARD (one per column)."""

    def __init__(self, lengthscales: float | np.ndarray = 1.0) -> None:
        self.lengthscales = np.asarray(lengthscales, dtype=float)
        if np.any(self.lengthscales <= 0):
            raise ValueError("lengthscales must be positive")

    def K(self, X: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * square_distance(X / self.lengthscales, X2 / self.lengthscales))


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X
