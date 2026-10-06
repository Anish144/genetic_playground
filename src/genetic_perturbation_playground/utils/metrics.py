import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error over all perturbations and genes."""
    return float(np.mean((pred - true) ** 2))


def pearson_per_row(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Pearson correlation across genes, one value per perturbation (row)."""
    p = pred - pred.mean(axis=1, keepdims=True)
    t = true - true.mean(axis=1, keepdims=True)
    return (p * t).sum(axis=1) / np.sqrt((p**2).sum(axis=1) * (t**2).sum(axis=1))


def pearson_top_k(pred: np.ndarray, true: np.ndarray, k: int = 20) -> np.ndarray:
    """Per-perturbation Pearson restricted to the k genes with the largest |true| effect."""
    idx = np.argsort(-np.abs(true), axis=1)[:, :k]
    return pearson_per_row(np.take_along_axis(pred, idx, 1), np.take_along_axis(true, idx, 1))
