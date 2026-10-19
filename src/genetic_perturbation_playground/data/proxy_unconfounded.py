"""Synthetic multi-environment regression data for proxy transfer without X-U confounding.

Within environment e, (X, Y) pairs and proxies W are observed in separate, unpaired samples.
The latent has ``dim_u`` = k dimensions:

    theta_e      ~ Uniform(-2, 2)^k                     once per environment
    U1_ei, U2_ej ~ N(theta_e, I_k)                       same law, drawn independently
    X_ei         ~ N(0, 1)                               independent of E and U
    Y_ei         = f(X_ei, U1_ei) + N(0, 0.1^2),         f(x, u) = sum_j f1(x, u_j) / sqrt(k)
    W_ej         = g(U2_ej) + N(0, 0.5^2 I_k),           g from PROXIES, applied per coordinate

with f1(x, u) = tanh(x u) + sin(u) / 2. The 1 / sqrt(k) keeps the spread of E[Y | x, e] across
environments comparable as k grows; k = 1 is the original one-dimensional setting.

In the target environment only W is observed; the transfer target is
E[Y | x, e*] = sum_j ∫ f1(x, u) N(u; theta_e*_j, 1) du / sqrt(k), exact via ``conditional_mean``.

Identification needs W to determine the law of U, i.e. g injective. "square" breaks this
(theta and -theta give the same law of W) and serves as a negative control.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

PROXIES = {
    "identity": lambda u: u,
    "u_plus_sin": lambda u: u + np.sin(u),  # injective, with flat points
    "tanh": np.tanh,  # injective, saturates for |u| > ~1.5
    "square": np.square,  # not injective
}


@dataclass(frozen=True)
class UnconfoundedSample:
    """One environment; (x, y, u1) are paired, (w, u2) are a separate sample."""

    theta: np.ndarray  # (k,)
    x: np.ndarray  # (n,)
    y: np.ndarray  # (n,)
    w: np.ndarray  # (m, k)
    u1: np.ndarray  # (n, k), latent
    u2: np.ndarray  # (m, k), latent


def outcome_mean(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """f(x, u) = E[Y | x, U1 = u] for u of shape (..., k), broadcasting x against u[..., 0]."""
    x = np.asarray(x)[..., None]
    return np.sum(np.tanh(x * u) + 0.5 * np.sin(u), axis=-1) / np.sqrt(u.shape[-1])


def conditional_mean(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """E[Y | x, e] = E_{U ~ N(theta, I)}[f(x, U)]; f is additive in U, so 1-D quadrature per j."""
    nodes, weights = hermegauss(60)
    f = outcome_mean(np.asarray(x)[:, None], theta + nodes[:, None])  # (n, nodes)
    return f @ weights / weights.sum()


def sample_environment(
    theta: np.ndarray,
    n: int,
    m: int,
    rng: np.random.Generator,
    proxy: str = "identity",
) -> UnconfoundedSample:
    """Draw ``n`` (X, Y) pairs and ``m`` proxies W from the environment with mean ``theta``."""
    k = len(theta)
    u1 = theta + rng.standard_normal((n, k))
    u2 = theta + rng.standard_normal((m, k))
    x = rng.standard_normal(n)
    y = outcome_mean(x, u1) + 0.1 * rng.standard_normal(n)
    w = PROXIES[proxy](u2) + 0.5 * rng.standard_normal((m, k))
    return UnconfoundedSample(theta=theta, x=x, y=y, w=w, u1=u1, u2=u2)


def sample_environments(
    d: int,
    n: int,
    m: int,
    rng: np.random.Generator,
    dim_u: int = 1,
    proxy: str = "identity",
) -> list[UnconfoundedSample]:
    """Draw ``d`` environments with theta_e ~ Uniform(-2, 2)^dim_u."""
    thetas = rng.uniform(-2.0, 2.0, size=(d, dim_u))
    return [sample_environment(theta, n, m, rng, proxy=proxy) for theta in thetas]
