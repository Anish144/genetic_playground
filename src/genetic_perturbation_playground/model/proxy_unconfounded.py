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


def mean_inner(kernel_w: Kernel, ws: list[np.ndarray], ws2: list[np.ndarray] | None = None):
    """<mu_a, mu_b> = mean of L(w_a, w_b) for every pair of environments, shape (len(ws), len(ws2))."""
    ws2 = ws if ws2 is None else ws2
    return np.array([[kernel_w(a, b).mean() for b in ws2] for a in ws])


class MarginalEmbeddingBridge:
    """Bridge h(x, w) estimated from D source environments.

    With M[e, e'] = <mu_e, mu_e'> = mean of L(W_e, W_e') and env(i) the environment of sample i,

        G = K_X ⊙ M[env, env],   c = (G + N * reg * I)^{-1} y,
        prediction(x) = sum_i c_i k(x_i, x) <mu_{env(i)}, mu_{e*}>.

    ``fit``/``predict`` take proxy samples; ``fit_embedded``/``predict_embedded`` take the inner
    products M directly, so callers can compute them once and reuse them across bridges.
    """

    def __init__(self, kernel_x: Kernel, kernel_w: Kernel, reg: float = 1e-3) -> None:
        self.kernel_x = kernel_x
        self.kernel_w = kernel_w
        self.reg = reg

    def fit(
        self, xs: list[np.ndarray], ys: list[np.ndarray], ws: list[np.ndarray]
    ) -> "MarginalEmbeddingBridge":
        """Fit from per-environment (x, y) pairs ``xs``, ``ys`` and unpaired proxies ``ws``."""
        self.w_envs_ = ws
        env = np.repeat(np.arange(len(xs)), [len(xi) for xi in xs])
        return self.fit_embedded(
            np.concatenate(xs), np.concatenate(ys), env, mean_inner(self.kernel_w, ws)
        )

    def predict(self, x: np.ndarray, w_target: np.ndarray) -> np.ndarray:
        """E[Y | x, e*] for the environment whose proxy sample is ``w_target``."""
        return self.predict_embedded(x, mean_inner(self.kernel_w, self.w_envs_, [w_target])[:, 0])

    def fit_embedded(
        self,
        x: np.ndarray,
        y: np.ndarray,
        env: np.ndarray,
        inner: np.ndarray,
        counts: np.ndarray | None = None,
    ) -> "MarginalEmbeddingBridge":
        """Stage 2 given ``inner[e, e'] = <mu_e, mu_e'>``; ``env`` indexes rows of ``inner``.

        Sample i may stand for ``counts[i]`` replicates sharing (x_i, env_i), with y_i their mean:
        the same fit as on the replicates, with ridge N * reg * diag(1 / counts), N = sum(counts).
        """
        counts = np.ones(len(x)) if counts is None else counts
        gram = self.kernel_x(x) * inner[np.ix_(env, env)]
        ridge = np.diag(self.reg * counts.sum() / counts)
        self.coef_ = cho_solve(cho_factor(gram + ridge), y)
        self.x_train_, self.env_ = x, env
        return self

    def predict_embedded(self, x: np.ndarray, inner_target: np.ndarray) -> np.ndarray:
        """Prediction given ``inner_target[e] = <mu_e, mu_{e*}>`` for every training environment."""
        return (self.kernel_x(self.x_train_, x) * inner_target[self.env_, None]).T @ self.coef_
