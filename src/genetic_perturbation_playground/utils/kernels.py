"""NumPy kernels with the same class hierarchy as GPflow (gpflow/kernels/{base,stationaries}.py).

    Kernel                      K(X, X2)
    └── Stationary              variance, lengthscales (scalar, or one per column for ARD)
        └── IsotropicStationary depends only on r = ‖(x - x') / ℓ‖; implement K_r2 or K_r
            └── RBF

A new isotropic kernel only needs ``K_r2`` (e.g. rational quadratic) or ``K_r`` (e.g. Matérn).
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


class Stationary(Kernel):
    """Kernel depending only on x - x'. An array of lengthscales gives ARD (one per column)."""

    def __init__(self, variance: float = 1.0, lengthscales: float | np.ndarray = 1.0) -> None:
        self.lengthscales = np.asarray(lengthscales, dtype=float)
        if variance <= 0 or np.any(self.lengthscales <= 0):
            raise ValueError("variance and lengthscales must be positive")
        self.variance = float(variance)

    def scale(self, X: np.ndarray) -> np.ndarray:
        return X / self.lengthscales


class IsotropicStationary(Stationary):
    """Kernel depending only on r = ‖(x - x') / ℓ‖. Subclasses implement K_r2 or K_r."""

    def K(self, X: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.K_r2(square_distance(self.scale(X), self.scale(X2)))

    def K_r2(self, r2: np.ndarray) -> np.ndarray:
        return self.K_r(np.sqrt(r2))

    def K_r(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} must implement K_r2 or K_r")


class RBF(IsotropicStationary):
    """k(x, x') = σ² exp(-‖x - x'‖² / (2ℓ²)), a.k.a. the squared exponential / Gaussian kernel."""

    def K_r2(self, r2: np.ndarray) -> np.ndarray:
        return self.variance * np.exp(-0.5 * r2)


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X
