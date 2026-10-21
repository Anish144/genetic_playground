"""
Neural counterpart of MarginalEmbeddingBridge: learned feature maps in place of RBF features.

Notation:
    D         number of source environments, indexed by e
    n_e, m_e  (X, Y) pairs and proxies W observed in environment e (unpaired samples)
    N         total (X, Y) pairs, sum_e n_e
    d_x, d_w  dimensions of X and W (1-D inputs are treated as one column)
    p, q      feature dimensions of phi and psi: ``n_features`` and ``n_features_w``

h(x, w) = phi(x)^T A psi(w) with MLPs phi: R^d_x -> R^p, psi: R^d_w -> R^q and A in R^{p x q}, so
E[Y | x, e] = phi(x)^T A mu_e with mu_e = E[psi(W) | e] in R^q (linear in mu_e, as
identification needs).

Stage 1: mu_e = mean_j psi(w_ej), the closed-form embedding of each environment (X ⊥ (U, E)
means no regression on x is needed). Stage 2: A = ridge regression of y on the p q features
phi(x_i) ⊗ mu_env(i), also closed form. phi and psi are trained with AdamW on the stage-2 squared
error, backpropagating through A and mu_e (as in DFPV, Xu et al., 2021).
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from tqdm import trange


def _mlp(d_in: int, width: int, depth: int, d_out: int) -> nn.Sequential:
    """ReLU MLP with `depth` hidden layers of `width` units, in float64."""
    dims = [d_in] + [width] * depth
    hidden = [layer for a, b in pairwise(dims) for layer in (nn.Linear(a, b), nn.ReLU())]
    return nn.Sequential(*hidden, nn.Linear(dims[-1], d_out)).double()


def _tensor(a: np.ndarray, device: str) -> torch.Tensor:
    """(n,) or (n, d) array -> float64 tensor of shape (n, d) on ``device``."""
    return torch.as_tensor(np.asarray(a, dtype=np.float64), device=device).reshape(len(a), -1)


class _Stage2Data(NamedTuple):
    """Training tensors: pooled pairs (x, y, c, env) and per-environment proxies."""

    x: torch.Tensor  # (N, d_x)
    y: torch.Tensor  # (N,) or (N, d_y)
    c: torch.Tensor  # (N,) replicate counts
    env: torch.Tensor  # (N,) environment index of each pair
    w_envs: list[torch.Tensor]  # D tensors of shape (m_e, d_w)


def _features(phi_x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """phi_x: (N, p), mu: (N, q) or (1, q) -> rows phi(x_i) ⊗ mu_i, shape (N, p q)."""
    return (phi_x[:, :, None] * mu[:, None, :]).flatten(1)


def _weighted_ridge(feats: torch.Tensor, y: torch.Tensor, c: torch.Tensor, reg: float):
    """A minimising sum_i c_i |feats_i A - y_i|^2 + reg * sum(c) * |A|^2."""
    weighted = feats * c[:, None]
    eye = torch.eye(feats.shape[1], dtype=feats.dtype, device=feats.device)
    return torch.linalg.solve(weighted.T @ feats + reg * c.sum() * eye, weighted.T @ y)


def _weighted_mse(pred: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Count-weighted mean over pairs of the squared error averaged over outputs."""
    sq_err = ((pred - y) ** 2).reshape(len(c), -1).mean(dim=1)
    return (c * sq_err).sum() / c.sum()


@dataclass
class NeuralMarginalEmbeddingBridge:
    """Bridge h(x, w) = phi(x)^T A psi(w) learned from D source environments.

    n_features  p, output dimension of phi
    n_features_w  q, output dimension of psi
    width       hidden units per layer of each MLP
    depth       hidden layers per MLP
    reg         ridge penalty lambda on A, scaled by N as in the kernel bridge
    lr          AdamW learning rate for the parameters of phi and psi
    weight_decay  AdamW decoupled weight decay on phi and psi (0 is plain Adam)
    n_steps     full-batch AdamW steps
    device      torch device to train and predict on, e.g. "cpu" or "cuda" (not "mps": the model
                is float64, which the Apple backend does not support)

    After ``fit``: ``phi_``, ``psi_`` (the MLPs) and ``A_``, A flattened to shape (p q,).
    """

    n_features: int = 16
    n_features_w: int = 16
    width: int = 64
    depth: int = 2
    reg: float = 1e-3
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_steps: int = 2000
    seed: int = 0
    device: str = "cpu"

    def fit(
        self,
        xs: list[np.ndarray],
        ys: list[np.ndarray],
        ws: list[np.ndarray],
        counts: list[np.ndarray] | None = None,
        checkpoints: Collection[int] = (),
        on_checkpoint: Callable[[int], None] | None = None,
    ) -> "NeuralMarginalEmbeddingBridge":
        """Fit from D source environments; element e of each list belongs to environment e.

        xs[e]: (n_e,) or (n_e, d_x);  ys[e]: (n_e,) or (n_e, d_y);  ws[e]: (m_e,) or (m_e, d_w).
        counts[e]: (n_e,), optional; pair i stands for counts[e][i] replicates with mean y_i, so the
        fit equals the fit on the replicates (weighted loss, ridge scaled by their total).
        After each step in ``checkpoints``, A_ is set and ``on_checkpoint(step)`` is called, so the
        bridge can be scored there as if fit with n_steps=step.
        """
        torch.manual_seed(self.seed)
        data = self._to_tensors(xs, ys, ws, counts)
        self._build_networks(data.x.shape[1], data.w_envs[0].shape[1])
        self._train(data, checkpoints, on_checkpoint)
        self._solve_A(data)
        return self

    def predict(self, x: np.ndarray, w_target: np.ndarray) -> np.ndarray:
        """E[Y | x, e*] for the target environment whose proxies are ``w_target``.

        x: (n_new,) or (n_new, d_x);  w_target: (m*,) or (m*, d_w);  returns (n_new,) or (n_new, d_y).
        """
        with torch.no_grad():
            mu = self.psi_(_tensor(w_target, self.device)).mean(dim=0, keepdim=True)  # (1, q)
            phi_x = self.phi_(_tensor(x, self.device))
            return (_features(phi_x, mu) @ self.A_).cpu().numpy()

    def _to_tensors(self, xs, ys, ws, counts) -> _Stage2Data:
        """Pool the environments' pairs into tensors on the device."""
        sizes = [len(xi) for xi in xs]
        c = np.ones(sum(sizes)) if counts is None else np.concatenate(counts)
        return _Stage2Data(
            x=_tensor(np.concatenate(xs), self.device),
            y=torch.as_tensor(np.concatenate(ys), dtype=torch.float64, device=self.device),
            c=torch.as_tensor(c, dtype=torch.float64, device=self.device),
            env=torch.as_tensor(np.repeat(np.arange(len(xs)), sizes), device=self.device),
            w_envs=[_tensor(w, self.device) for w in ws],
        )

    def _build_networks(self, d_x: int, d_w: int) -> None:
        """Freshly initialised phi and psi on the device."""
        self.phi_ = _mlp(d_x, self.width, self.depth, self.n_features).to(self.device)
        self.psi_ = _mlp(d_w, self.width, self.depth, self.n_features_w).to(self.device)

    def _train(self, data: _Stage2Data, checkpoints, on_checkpoint) -> None:
        """Full-batch AdamW on phi and psi against the stage-2 loss."""
        params = [*self.phi_.parameters(), *self.psi_.parameters()]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        progress = trange(self.n_steps, desc="neural bridge", leave=False)
        for step in progress:
            optimizer.zero_grad()
            loss, _ = self._stage2(data)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step + 1 in checkpoints:
                self._solve_A(data)
                on_checkpoint(step + 1)

    def _solve_A(self, data: _Stage2Data) -> None:
        """Set A_ to the ridge solution for the current phi and psi."""
        with torch.no_grad():
            _, self.A_ = self._stage2(data)

    def _stage2(self, data: _Stage2Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form weighted ridge A for the current features, and its weighted squared error.

        Returns the scalar loss and A, shape (p q,) or (p q, d_y).
        """
        mu = self._embed(data.w_envs)  # (D, q), stage 1
        feats = _features(self.phi_(data.x), mu[data.env])  # (N, p q)
        A = _weighted_ridge(feats, data.y, data.c, self.reg)
        return _weighted_mse(feats @ A, data.y, data.c), A

    def _embed(self, w_envs: list[torch.Tensor]) -> torch.Tensor:
        """Mean psi features of each environment's proxies, shape (D, q)."""
        return torch.stack([self.psi_(w).mean(dim=0) for w in w_envs])
