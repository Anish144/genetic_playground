"""
Kernel bridge for proxy transfer when X is independent of the latent and the environment.

E[Y | x, e] = <h, phi(x) ⊗ mu_e> with mu_e = E[psi(W) | e], the marginal mean embedding of W
in environment e. Because X is independent of (U, E), stage 1 needs no regression on X:
mu_e = (1 / m_e) sum_j psi(w_ej). Stage 2 is kernel ridge regression of y on
v_i = phi(x_i) ⊗ mu_{e_i}, and the target prediction is <h, phi(x) ⊗ mu_{e*}>.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from genetic_perturbation_playground.utils.kernels import Kernel


class MarginalEmbeddingBridge:
    """Bridge h(x, w) estimated from D source environments.

    With M[e, e'] = <mu_e, mu_e'> = mean of L(W_e, W_e') and env(i) the environment of sample i,

        G = K_X ⊙ M[env, env],   c = (G + N * reg * I)^{-1} y,
        prediction(x) = sum_i c_i k(x_i, x) <mu_{env(i)}, mu_{e*}>.
    """

    def __init__(self, kernel_x: Kernel, kernel_w: Kernel, reg: float = 1e-3) -> None:
        self.kernel_x = kernel_x
        self.kernel_w = kernel_w
        self.reg = reg

        self.x_train_ = None
        self.env_ = None
        self.w_envs_ = None
        self.coef_ = None

    def fit(
        self, xs: list[np.ndarray], ys: list[np.ndarray], ws: list[np.ndarray]
    ) -> "MarginalEmbeddingBridge":
        """Fit from per-environment (x, y) pairs ``xs``, ``ys`` and unpaired proxies ``ws``."""
        x, y = np.concatenate(xs), np.concatenate(ys)
        env = np.repeat(np.arange(len(xs)), [len(xi) for xi in xs])
        self.w_envs_ = ws

        n = len(x)
        mean_inner = self._mean_inner(ws)  # (D, D)
        gram = self.kernel_x(x) * mean_inner[np.ix_(env, env)]
        self.coef_ = cho_solve(cho_factor(gram + self.reg * n * np.eye(n)), y)
        self.x_train_, self.env_ = x, env
        return self

    def predict(self, x: np.ndarray, w_target: np.ndarray) -> np.ndarray:
        """E[Y | x, e*] for the environment whose proxy sample is ``w_target``."""
        mean_inner = self._mean_inner(self.w_envs_, [w_target])[:, 0]  # (D,)
        return self.coef_ @ (self.kernel_x(self.x_train_, x) * mean_inner[self.env_, None])

    def _mean_inner(self, ws: list[np.ndarray], ws2: list[np.ndarray] | None = None) -> np.ndarray:
        """<mu_a, mu_b> = mean of L(w_a, w_b) for every pair of environments."""
        ws2 = ws if ws2 is None else ws2
        return np.array([[self.kernel_w(a, b).mean() for b in ws2] for a in ws])
