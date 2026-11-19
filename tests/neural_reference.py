"""Small, independent dense references for testing the optimized neural ridge solver."""

from types import SimpleNamespace

import numpy as np
import torch


def pair_data(xs, ys, ws, counts=None, device="cpu"):
    sizes = [len(x) for x in xs]
    return SimpleNamespace(
        x=torch.as_tensor(np.concatenate(xs), dtype=torch.float64, device=device),
        y=torch.as_tensor(np.concatenate(ys), dtype=torch.float64, device=device),
        c=torch.as_tensor(
            np.ones(sum(sizes)) if counts is None else np.concatenate(counts),
            dtype=torch.float64,
            device=device,
        ),
        env=torch.as_tensor(np.repeat(np.arange(len(xs)), sizes), device=device),
        w_envs=[torch.as_tensor(w, dtype=torch.float64, device=device) for w in ws],
    )


def features(categories, embeddings):
    """Explicit category-by-feature design, deliberately dense for independent tests."""
    return (categories[:, :, None] * embeddings[:, None, :]).flatten(1)


def weighted_ridge(design, targets, counts, reg):
    gram = design.T @ (counts[:, None] * design)
    penalty = reg * counts.sum() * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + penalty, design.T @ (counts[:, None] * targets))


def weighted_sse(prediction, targets, counts):
    return (counts * (prediction - targets).square().reshape(len(counts), -1).mean(dim=1)).sum()
