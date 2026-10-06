import numpy as np
import pandas as pd


def mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean squared error across genes, one value per perturbation (row)."""
    return np.mean((pred - true) ** 2, axis=1)


def pearson_per_row(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Pearson correlation across genes, one value per perturbation (row)."""
    p = pred - pred.mean(axis=1, keepdims=True)
    t = true - true.mean(axis=1, keepdims=True)
    return (p * t).sum(axis=1) / np.sqrt((p**2).sum(axis=1) * (t**2).sum(axis=1))


def pearson_top_k(pred: np.ndarray, true: np.ndarray, k: int = 20) -> np.ndarray:
    """Per-perturbation Pearson restricted to the k genes with the largest |true| effect."""
    idx = np.argsort(-np.abs(true), axis=1)[:, :k]
    return pearson_per_row(np.take_along_axis(pred, idx, 1), np.take_along_axis(true, idx, 1))


def score(preds: dict[str, np.ndarray], true: np.ndarray, perturbations, k: int) -> pd.DataFrame:
    """Long table (perturbation, metric, model, value) of per-perturbation scores for each model.

    MSE also gets a "no effect" baseline (predicting a zero effect).
    """
    scores = {}
    for name, p in preds.items():
        scores[("mse", name)] = mse(p, true)
        scores[("pearson", name)] = pearson_per_row(p, true)
        scores[(f"pearson_top{k}", name)] = pearson_top_k(p, true, k)
    scores[("mse", "no effect")] = mse(np.zeros_like(true), true)
    table = pd.DataFrame(scores, index=pd.Index(perturbations, name="perturbation"))
    return table.melt(ignore_index=False, var_name=["metric", "model"]).reset_index()
