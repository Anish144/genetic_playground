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

NeuralPerXEmbeddingBridge is the special case phi(x) = x for one-hot x: a separate A_x per
category, h(x, w) = A_x^T psi(w), so only psi is shared across categories.
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import trange


class _SwiGLU(nn.Module):
    """Layer h -> silu(h W_gate + b_gate) * (h W_value + b_value), from d_in to d_out units."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.gate = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(self.gate(h)) * self.value(h)


class _Residual(nn.Module):
    """h -> h + layer(h)."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.layer(h)


def _mlp(d_in: int, width: int, depth: int, d_out: int) -> nn.Sequential:
    """MLP with `depth` SwiGLU hidden layers of `width` units and a linear output, in float64.
    Hidden layers whose input has `width` units too (all but the first) are residual."""
    dims = [d_in] + [width] * depth
    hidden = [_Residual(_SwiGLU(a, b)) if a == b else _SwiGLU(a, b) for a, b in pairwise(dims)]
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


def _normal_equations(feats: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> tuple:
    """Weighted Gram F^T C F and cross term F^T C y of a weighted least-squares fit."""
    weighted = feats * c[:, None]
    return weighted.T @ feats, weighted.T @ y


def _ridge_solve(gram: torch.Tensor, cross: torch.Tensor, weight: torch.Tensor, reg: float):
    """Ridge solution from the normal equations, with penalty reg * weight * |A|^2."""
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + reg * weight * eye, cross)


def _weighted_ridge(feats: torch.Tensor, y: torch.Tensor, c: torch.Tensor, reg: float):
    """A minimising sum_i c_i |feats_i A - y_i|^2 + reg * sum(c) * |A|^2."""
    return _ridge_solve(*_normal_equations(feats, y, c), c.sum(), reg)


def _weighted_sse(pred: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Count-weighted sum over pairs of the squared error averaged over outputs."""
    sq_err = ((pred - y) ** 2).reshape(len(c), -1).mean(dim=1)
    return (c * sq_err).sum()


@dataclass
class NeuralMarginalEmbeddingBridge:
    """Bridge h(x, w) = phi(x)^T A psi(w) learned from D source environments.

    n_features  p, output dimension of phi
    n_features_w  q, output dimension of psi
    width_x     hidden units per layer of phi
    width_w     hidden units per layer of psi
    depth       hidden layers per MLP
    reg         ridge penalty lambda on A, scaled by N as in the kernel bridge
    lr          AdamW learning rate for the parameters of phi and psi
    weight_decay  AdamW decoupled weight decay on phi and psi (0 is plain Adam)
    n_steps     full-batch AdamW steps
    cross_fit_folds  0 trains phi and psi on the in-sample stage-2 loss. K >= 2 trains them on a
                cross-fitted loss: each step splits the environments into K random folds (at most
                one per environment), solves A on all folds but one and scores the held-out fold,
                for every fold. The final A is still solved on all environments.
    device      torch device to train and predict on, e.g. "cpu" or "cuda" (not "mps": the model
                is float64, which the Apple backend does not support)
    log_key     if set and a W&B run is open, each step logs <log_key>/loss and <log_key>/step

    After ``fit``: ``phi_``, ``psi_`` (the MLPs) and ``A_``, A flattened to shape (p q,).
    """

    n_features: int = 16
    n_features_w: int = 16
    width_x: int = 64
    width_w: int = 64
    depth: int = 2
    reg: float = 1e-3
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_steps: int = 2000
    cross_fit_folds: int = 0
    seed: int = 0
    device: str = "cpu"
    log_key: str | None = None

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
        self.phi_ = _mlp(d_x, self.width_x, self.depth, self.n_features).to(self.device)
        self.psi_ = _mlp(d_w, self.width_w, self.depth, self.n_features_w).to(self.device)

    def _train(self, data: _Stage2Data, checkpoints, on_checkpoint) -> None:
        """Full-batch AdamW on phi and psi against the stage-2 loss."""
        params = [*self.phi_.parameters(), *self.psi_.parameters()]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        logging = self.log_key and wandb.run is not None  # the caller's run, if it opened one
        if logging:  # the loss against this fit's own step
            wandb.define_metric(f"{self.log_key}/*", step_metric=f"{self.log_key}/step")
        progress = trange(self.n_steps, desc="neural bridge", leave=False)
        for step in progress:
            optimizer.zero_grad()
            loss = self._training_loss(data)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if logging:
                wandb.log({f"{self.log_key}/loss": loss.item(), f"{self.log_key}/step": step})
            if step + 1 in checkpoints:
                self._solve_A(data)
                on_checkpoint(step + 1)

    def _training_loss(self, data: _Stage2Data) -> torch.Tensor:
        """The loss phi and psi are trained on: cross-fitted, or in-sample stage 2."""
        if self.cross_fit_folds:
            return self._cross_fit_loss(data)
        loss, _ = self._stage2(data)
        return loss

    def _cross_fit_loss(self, data: _Stage2Data) -> torch.Tensor:
        """Weighted squared error of each fold's pairs under A solved on the other folds."""
        mu = self._embed(data.w_envs)
        feats = _features(self.phi_(data.x), mu[data.env])
        fold = self._random_folds(data)
        held_out = [fold == k for k in range(int(fold.max()) + 1)]
        parts = [_normal_equations(feats[h], data.y[h], data.c[h]) for h in held_out]
        gram, cross = map(sum, zip(*parts, strict=True))
        sse = sum(
            self._held_out_sse(feats, data, h, gram - g, cross - b)
            for h, (g, b) in zip(held_out, parts, strict=True)
        )
        return sse / data.c.sum()

    def _held_out_sse(self, feats, data: _Stage2Data, held, gram, cross) -> torch.Tensor:
        """Weighted sum of squared errors of the `held` pairs, under A from the others' equations."""
        A = _ridge_solve(gram, cross, data.c[~held].sum(), self.reg)
        return _weighted_sse(feats[held] @ A, data.y[held], data.c[held])

    def _random_folds(self, data: _Stage2Data) -> torch.Tensor:
        """Fold of each pair: its environment's fold in a random split of the environments."""
        return self._env_folds(len(data.w_envs))[data.env]

    def _env_folds(self, n_envs: int) -> torch.Tensor:
        """Fold of each environment in a random split into cross_fit_folds (at most n_envs)."""
        n_folds = min(self.cross_fit_folds, n_envs)
        if n_folds < 2:
            raise ValueError("cross-fitting needs at least two folds and two environments")
        env_fold = torch.randperm(n_envs) % n_folds  # balanced: fold sizes differ by at most one
        return env_fold.to(self.device)

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
        return _weighted_sse(feats @ A, data.y, data.c) / data.c.sum(), A

    def _embed(self, w_envs: list[torch.Tensor]) -> torch.Tensor:
        """Mean psi features of each environment's proxies, shape (D, q)."""
        return torch.stack([self.psi_(w).mean(dim=0) for w in w_envs])


class _GroupedData(NamedTuple):
    """Sums of the pairs per (category, environment), and per-environment proxies."""

    c: torch.Tensor  # (G, D) replicate counts
    cy: torch.Tensor  # (G, D, d_y) count-weighted sums of y
    cyy: torch.Tensor  # (D,) count-weighted sums of |y|^2 / d_y
    w_envs: list[torch.Tensor]  # D tensors of shape (m_e, d_w)


def _sums(data: _GroupedData, mu: torch.Tensor, envs: torch.Tensor) -> tuple:
    """Per-category Gram (G, q, q) and cross term (G, q, d_y), sum of c |y|^2 / d_y and total
    count over the pairs of the environments in the mask `envs` (D,)."""
    c = data.c * envs
    gram = (c[:, :, None] * mu).transpose(1, 2) @ mu
    cross = (mu * envs[:, None]).T @ data.cy
    return gram, cross, data.cyy[envs].sum(), c.sum()


def _sse(A: torch.Tensor, gram: torch.Tensor, cross: torch.Tensor, cyy: torch.Tensor):
    """sum_i c_i |y_i - A_x(i)^T mu_i|^2 / d_y over the pairs that `gram`, `cross`, `cyy` sum."""
    return cyy - (2 * (A * cross).sum() - (A * (gram @ A)).sum()) / cross.shape[-1]


@dataclass
class NeuralPerXEmbeddingBridge(NeuralMarginalEmbeddingBridge):
    """Bridge h(x, w) = A_x^T psi(w) with a separate A_x in R^{q x d_y} per one-hot category x.

    NeuralMarginalEmbeddingBridge with phi(x) = x: no response surface is shared across categories,
    only psi. A_x is the ridge regression of y on mu_e over x's pairs alone, with the same penalty
    reg * N. n_features, width_x are unused; ys must be 2-D. The losses are computed from sums per
    (category, environment), never per pair: memory and time grow as G q d_y, not (G q)^2.

    After ``fit``: ``psi_`` and ``A_``, shape (G, q, d_y); ``phi_`` is the identity.
    """

    def fit(self, xs, ys, ws, counts=None, checkpoints=(), on_checkpoint=None):
        """As NeuralMarginalEmbeddingBridge.fit; each row of xs[e] is one-hot."""
        torch.manual_seed(self.seed)
        data = self._to_tensors(xs, ys, ws, counts)
        self.phi_ = nn.Identity()
        self.psi_ = _mlp(data.w_envs[0].shape[1], self.width_w, self.depth, self.n_features_w)
        self.psi_.to(self.device)
        self._train(data, checkpoints, on_checkpoint)
        self._solve_A(data)
        return self

    def predict(self, x: np.ndarray, w_target: np.ndarray) -> np.ndarray:
        """E[Y | x, e*], shape (n_new, d_y), for one-hot x and the target's proxies ``w_target``."""
        with torch.no_grad():
            mu = self.psi_(_tensor(w_target, self.device)).mean(dim=0)  # (q,)
            pred = mu @ self.A_  # (G, d_y), every category's prediction
            return pred[_tensor(x, self.device).argmax(dim=1)].cpu().numpy()

    def _to_tensors(self, xs, ys, ws, counts) -> _GroupedData:
        """The pairs summed per (category, environment), on the device."""
        pairs = super()._to_tensors(xs, ys, ws, counts)
        n_cats, n_envs = pairs.x.shape[1], len(ws)
        cell = pairs.x.argmax(dim=1) * n_envs + pairs.env  # flat (category, environment) index

        def summed(values, index, n):
            zeros = torch.zeros(n, *values.shape[1:], dtype=values.dtype, device=values.device)
            return zeros.index_add(0, index, values)

        n = n_cats * n_envs
        return _GroupedData(
            c=summed(pairs.c, cell, n).reshape(n_cats, n_envs),
            cy=summed(pairs.c[:, None] * pairs.y, cell, n).reshape(n_cats, n_envs, -1),
            cyy=summed(pairs.c * (pairs.y**2).mean(dim=1), pairs.env, n_envs),
            w_envs=pairs.w_envs,
        )

    def _stage2(self, data: _GroupedData) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form ridge A_x of every category, and the weighted squared error."""
        mu = self._embed(data.w_envs)
        gram, cross, cyy, n = _sums(data, mu, torch.ones(len(mu), dtype=bool, device=mu.device))
        A = _ridge_solve(gram, cross, n, self.reg)
        return _sse(A, gram, cross, cyy) / n, A

    def _cross_fit_loss(self, data: _GroupedData) -> torch.Tensor:
        """Weighted squared error of each fold's pairs under A_x solved on the other folds."""
        mu = self._embed(data.w_envs)
        fold = self._env_folds(len(mu))
        parts = [_sums(data, mu, fold == k) for k in range(int(fold.max()) + 1)]
        gram, cross, _, n = map(sum, zip(*parts, strict=True))
        sse = sum(
            _sse(_ridge_solve(gram - g, cross - b, n - m, self.reg), g, b, yy)
            for g, b, yy, m in parts
        )
        return sse / n
