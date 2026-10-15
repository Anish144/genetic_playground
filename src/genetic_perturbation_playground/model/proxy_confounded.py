"""
Model for the proxy multi-domain classification data.

This requires a 2 stage procedure:
- Find the conditional mean embedding \\mu_W|x,e using the solution to RKHS
regression.
- Plug this into the solution for the ridge regression of y on
<m, \\mu_W|x,e \\otimes \\phi(x)>.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from genetic_perturbation_playground.utils.kernels import Kernel


class ConditionalMeanEmbedding:
    """Conditional mean embedding of W | X within one domain (Grünewälder et al., 2012).

    Kernel ridge regression from X into the RKHS of W:

        mu_{W|x} = sum_i alpha_i(x) phi(w_i),   alpha(x) = (K_X + reg * n * I)^{-1} k_X(x).

    Only the kernel on X is needed to fit. The kernel on W enters when the embedding is
    evaluated, e.g. <mu_{W|x}, phi(w')> = k_W(w', W_train) @ alpha(x).
    """

    def __init__(self, kernel_x: Kernel, reg: float = 1e-3) -> None:
        self.kernel_x = kernel_x
        self.reg = reg

    def fit(self, x: np.ndarray, w: np.ndarray) -> "ConditionalMeanEmbedding":
        n = len(x)
        self.x_train_, self.w_train_ = x, w
        self.cholesky_factor_ = cho_factor(self.kernel_x(x) + self.reg * n * np.eye(n))
        return self

    def weights(self, x_new: np.ndarray) -> np.ndarray:
        """alpha(x_new), shape (n_train, n_new); column j holds the weights for x_new[j]."""
        return cho_solve(self.cholesky_factor_, self.kernel_x(self.x_train_, x_new))


class ConditionalEmbeddingBridge:
    """Kernel bridge function m_0 for multi-domain adaptation (Tsai et al., 2024, Sec. 4.3).

    Stage 1: a conditional mean embedding mu_e(x) of W | X for each source domain e.
    Stage 2: ridge regression of y on v_i = phi(x_i) ⊗ mu_{e_i}(x_i). With
    Gamma[k, i] = alpha_{e_i, k}(x_i) (zero unless sample k is from domain e_i),

        G = K_X ⊙ (Gamma^T L Gamma),   c = (G + N * reg_bridge * I)^{-1} y,
        m_0 = sum_i c_i sum_k Gamma[k, i] phi(x_i) ⊗ phi(w_k).

    The stages use disjoint random halves of the data, so that mu_{e_i}(x_i) is not fit to
    the w_i paired with y_i. Prediction under a domain with embedding mu(x) is
    <m_0, phi(x) ⊗ mu(x)>; pass the target-domain embedding to adapt.
    """

    def __init__(
        self,
        kernel_x: Kernel,
        kernel_w: Kernel,
        reg_cme: float = 1e-3,
        reg_bridge: float = 1e-3,
    ) -> None:
        self.kernel_x = kernel_x
        self.kernel_w = kernel_w
        self.reg_cme = reg_cme
        self.reg_bridge = reg_bridge

    def fit(
        self, x: np.ndarray, w: np.ndarray, y: np.ndarray, e: np.ndarray, rng: np.random.Generator
    ) -> "ConditionalEmbeddingBridge":
        """Fit on pooled source samples; ``e`` gives each sample's domain."""
        stage1 = rng.random(len(x)) < 0.5
        x1, w1, e1 = x[stage1], w[stage1], e[stage1]
        x2, y2, e2 = x[~stage1], y[~stage1], e[~stage1]

        # Stage 1: one embedding per domain, collected into a block-structured Gamma.
        gamma = np.zeros((len(x1), len(x2)))
        for domain in np.unique(e):
            rows, cols = np.flatnonzero(e1 == domain), np.flatnonzero(e2 == domain)
            cme = ConditionalMeanEmbedding(self.kernel_x, self.reg_cme).fit(x1[rows], w1[rows])
            gamma[np.ix_(rows, cols)] = cme.weights(x2[cols])

        # Stage 2: kernel ridge regression of y on v_i.
        n = len(x2)
        gram = self.kernel_x(x2) * (gamma.T @ self.kernel_w(w1) @ gamma)
        self.coef_ = cho_solve(cho_factor(gram + self.reg_bridge * n * np.eye(n)), y2)
        self.w_stage1_, self.x_stage2_, self.gamma_ = w1, x2, gamma
        return self

    def predict(self, x: np.ndarray, cme: ConditionalMeanEmbedding) -> np.ndarray:
        """<m_0, phi(x) ⊗ mu(x)> for the domain whose W | X embedding is ``cme``."""
        mean_inner = self.gamma_.T @ self.kernel_w(self.w_stage1_, cme.w_train_) @ cme.weights(x)
        return self.coef_ @ (self.kernel_x(self.x_stage2_, x) * mean_inner)
