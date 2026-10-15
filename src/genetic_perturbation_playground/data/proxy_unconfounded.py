"""Synthetic multi-environment regression data for proxy transfer without X-U confounding.

Within environment e, (X, Y) pairs and proxies W are observed in separate, unpaired samples:

    theta_e      ~ Uniform(theta_low, theta_high)       once per environment
    U1_ei, U2_ej ~ N(theta_e, 1)                         same law, drawn independently
    X_ei         ~ N(0, 1)                               independent of E and U
    Y_ei         = f(X_ei, U1_ei) + N(0, noise_y^2),     f(x, u) = tanh(x u) + sin(u) / 2
    W_ej         = g(U2_ej) + N(0, noise_w^2),           g chosen from PROXIES

In the target environment only W is observed; the transfer target is
E[Y | x, e*] = ∫ f(x, u) N(u; theta_e*, 1) du, available exactly via ``conditional_mean``.

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

    theta: float
    x: np.ndarray  # (n,)
    y: np.ndarray  # (n,)
    w: np.ndarray  # (m,)
    u1: np.ndarray  # (n,), latent
    u2: np.ndarray  # (m,), latent


def outcome_mean(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """f(x, u) = E[Y | x, U1 = u]."""
    return np.tanh(x * u) + 0.5 * np.sin(u)


def conditional_mean(x: np.ndarray, theta: float, n_nodes: int = 60) -> np.ndarray:
    """E[Y | x, e] = E_{U ~ N(theta, 1)}[f(x, U)], by Gauss-Hermite quadrature."""
    nodes, weights = hermegauss(n_nodes)
    f = outcome_mean(np.asarray(x)[:, None], theta + nodes[None, :])
    return f @ weights / weights.sum()


def sample_environment(
    theta: float,
    n: int,
    m: int,
    rng: np.random.Generator,
    noise_y: float = 0.1,
    noise_w: float = 0.5,
    proxy: str = "identity",
) -> UnconfoundedSample:
    """Draw ``n`` (X, Y) pairs and ``m`` proxies W from the environment with mean ``theta``."""
    u1 = theta + rng.standard_normal(n)
    u2 = theta + rng.standard_normal(m)
    x = rng.standard_normal(n)
    y = outcome_mean(x, u1) + noise_y * rng.standard_normal(n)
    w = PROXIES[proxy](u2) + noise_w * rng.standard_normal(m)
    return UnconfoundedSample(theta=theta, x=x, y=y, w=w, u1=u1, u2=u2)


def sample_environments(
    d: int,
    n: int,
    m: int,
    rng: np.random.Generator,
    theta_low: float = -2.0,
    theta_high: float = 2.0,
    proxy: str = "identity",
) -> list[UnconfoundedSample]:
    """Draw ``d`` environments with theta_e ~ Uniform(theta_low, theta_high)."""
    thetas = rng.uniform(theta_low, theta_high, size=d)
    return [sample_environment(theta, n, m, rng, proxy=proxy) for theta in thetas]
