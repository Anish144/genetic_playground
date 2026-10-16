"""
Neural counterpart of MarginalEmbeddingBridge: learned feature maps in place of RBF features.

Notation:
    D         number of source environments, indexed by e
    n_e, m_e  (X, Y) pairs and proxies W observed in environment e (unpaired samples)
    N         total (X, Y) pairs, sum_e n_e
    d_x, d_w  dimensions of X and W (1-D inputs are treated as one column)
    p, q      feature dimensions of phi and psi; both equal ``n_features``

h(x, w) = phi(x)^T A psi(w) with MLPs phi: R^d_x -> R^p, psi: R^d_w -> R^q and A in R^{p x q}, so
E[Y | x, e] = phi(x)^T A mu_e with mu_e = E[psi(W) | e] in R^q (linear in mu_e, as
identification needs).

Stage 1: mu_e = mean_j psi(w_ej), the closed-form embedding of each environment (X ⊥ (U, E)
means no regression on x is needed). Stage 2: A = ridge regression of y on the p q features
phi(x_i) ⊗ mu_env(i), also closed form. phi and psi are trained with Adam on the stage-2 squared
error, backpropagating through A and mu_e (as in DFPV, Xu et al., 2021).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import trange


def _mlp(d_in: int, width: int, depth: int, d_out: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out)).double()


def _tensor(a: np.ndarray) -> torch.Tensor:
    """(n,) or (n, d) array -> float64 tensor of shape (n, d)."""
    return torch.as_tensor(np.asarray(a, dtype=np.float64)).reshape(len(a), -1)


@dataclass
class NeuralMarginalEmbeddingBridge:
    """Bridge h(x, w) = phi(x)^T A psi(w) learned from D source environments.

    n_features  p = q, output dimension of phi and of psi
    width       hidden units per layer of each MLP
    depth       hidden layers per MLP
    reg         ridge penalty lambda on A, scaled by N as in the kernel bridge
    lr          Adam learning rate for the parameters of phi and psi
    n_steps     full-batch Adam steps

    After ``fit``: ``phi_``, ``psi_`` (the MLPs) and ``A_``, A flattened to shape (p q,).
    """

    n_features: int = 16
    width: int = 64
    depth: int = 2
    reg: float = 1e-3
    lr: float = 1e-3
    n_steps: int = 2000
    seed: int = 0

    def fit(
        self, xs: list[np.ndarray], ys: list[np.ndarray], ws: list[np.ndarray]
    ) -> "NeuralMarginalEmbeddingBridge":
        """Fit from D source environments; element e of each list belongs to environment e.

        xs[e]: (n_e,) or (n_e, d_x);  ys[e]: (n_e,);  ws[e]: (m_e,) or (m_e, d_w).
        """
        torch.manual_seed(self.seed)
        x = _tensor(np.concatenate(xs))  # (N, d_x)
        y = _tensor(np.concatenate(ys))[:, 0]  # (N,)
        env = torch.from_numpy(np.repeat(np.arange(len(xs)), [len(xi) for xi in xs]))  # (N,)
        w_envs = [_tensor(w) for w in ws]  # D tensors of shape (m_e, d_w)
        self.phi_ = _mlp(x.shape[1], self.width, self.depth, self.n_features)
        self.psi_ = _mlp(w_envs[0].shape[1], self.width, self.depth, self.n_features)

        params = [*self.phi_.parameters(), *self.psi_.parameters()]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        progress = trange(self.n_steps, desc="neural bridge", leave=False)
        for _ in progress:
            optimizer.zero_grad()
            loss, _ = self._stage2(x, y, env, w_envs)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        with torch.no_grad():
            _, self.A_ = self._stage2(x, y, env, w_envs)
        return self

    def predict(self, x: np.ndarray, w_target: np.ndarray) -> np.ndarray:
        """E[Y | x, e*] for the target environment whose proxies are ``w_target``.

        x: (n_new,) or (n_new, d_x);  w_target: (m*,) or (m*, d_w);  returns (n_new,).
        """
        with torch.no_grad():
            mu = self.psi_(_tensor(w_target)).mean(dim=0, keepdim=True)  # (1, q)
            return (self._features(self.phi_(_tensor(x)), mu) @ self.A_).numpy()

    def _stage2(self, x, y, env, w_envs) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form ridge A for the current features, and its mean squared error.

        x: (N, d_x), y: (N,), env: (N,) environment index of each pair, w_envs: D x (m_e, d_w).
        Returns the scalar loss and A flattened to (p q,).
        """
        mu = torch.stack([self.psi_(w).mean(dim=0) for w in w_envs])  # (D, q), stage 1
        feats = self._features(self.phi_(x), mu[env])  # (N, p q)
        n, d = feats.shape
        A = torch.linalg.solve(
            feats.T @ feats + self.reg * n * torch.eye(d, dtype=torch.float64), feats.T @ y
        )
        return torch.mean((feats @ A - y) ** 2), A

    @staticmethod
    def _features(phi_x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """phi_x: (N, p), mu: (N, q) or (1, q) -> rows phi(x_i) ⊗ mu_i, shape (N, p q)."""
        return (phi_x[:, :, None] * mu[:, None, :]).flatten(1)
