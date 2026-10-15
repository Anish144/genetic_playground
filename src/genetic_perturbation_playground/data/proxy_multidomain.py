"""Synthetic multi-domain classification data from Tsai et al., "Proxy Methods for Domain
Adaptation" (AISTATS 2024), Appendix D.3, originally from Alabdulmohsin et al. (2023).

Each domain differs only in the prior over the binary latent U:

    U         ~ Categorical(pi)
    W | U     ~ N(a_w o(U) M_{W|U}, 1)
    X | U     ~ N(o(U) M_{X|U}, I_2)
    C_i | X,U ~ Bernoulli(sigmoid([X M_{C|X,U} + o(U) M_{C|U}]_i)),   i = 1..3
    Y | C,U   ~ Bernoulli(sigmoid(C M_{Y|C,U} + o(U) M_{Y|U}))

where o(.) is the one-hot encoder. In the multi-domain setting C is latent; it is returned
only for diagnostics.

Parameters follow the reference implementation (``MultiWSimulator`` in
google-research/latent_shift_adaptation/data/lsa_synthetic.py), which differs from the
paper's text in two places: M_{Y|C,U} carries an extra factor of 2, and a_w scales the
W | U mean rather than the X | U mean. With these, oracle AUROCs match the paper's table.
"""

from dataclasses import dataclass, fields

import numpy as np

_M_W_U = np.array([-1.0, 1.0])
_M_X_U = np.array([[-1.0, 1.0], [1.0, -1.0]])
_M_C_U = np.array([[-2.0, 2.0, 2.0], [-1.0, 1.0, 2.0]])
_M_C_XU = 3.0 * np.array(  # indexed by u
    [
        [[-2.0, 2.0, -1.0], [1.0, -2.0, -3.0]],
        [[2.0, -2.0, 1.0], [-1.0, 2.0, 3.0]],
    ]
)
_M_Y_U = np.array([2.0, 2.0])
_M_Y_CU = 2.0 * np.array([[3.0, -2.0, -1.0], [3.0, -1.0, -2.0]])  # indexed by u

# P(U=0) of the source domains and the target domain for each task.
TASKS: dict[int, tuple[tuple[float, ...], float]] = {
    1: ((0.1, 0.2, 0.3), 0.9),
    2: ((0.4, 0.5, 0.6), 0.9),
    3: ((0.7, 0.8, 0.9), 0.4),
}


@dataclass(frozen=True)
class DomainSample:
    """Samples from one domain; ``z`` is the domain index, constant within a domain."""

    x: np.ndarray  # (n, 2) float
    w: np.ndarray  # (n,) float
    y: np.ndarray  # (n,) int
    z: np.ndarray  # (n,) int
    u: np.ndarray  # (n,) int, latent
    c: np.ndarray  # (n, 3) int, latent in the multi-domain setting

    @classmethod
    def concat(cls, samples: list["DomainSample"]) -> "DomainSample":
        names = [f.name for f in fields(cls)]
        return cls(**{n: np.concatenate([getattr(s, n) for s in samples]) for n in names})


def _bernoulli_logit(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)


def sample_domain(
    n: int, p_u0: float, z: int, rng: np.random.Generator, a_w: float = 1.0
) -> DomainSample:
    """Draw ``n`` samples from the domain with P(U=0) = ``p_u0``."""
    u = (rng.random(n) >= p_u0).astype(int)
    one_hot_u = np.eye(2)[u]

    w = a_w * (one_hot_u @ _M_W_U) + rng.standard_normal(n)
    x = one_hot_u @ _M_X_U + rng.standard_normal((n, 2))
    c = _bernoulli_logit(np.einsum("nd,ndk->nk", x, _M_C_XU[u]) + one_hot_u @ _M_C_U, rng)
    y = _bernoulli_logit(np.einsum("nk,nk->n", c, _M_Y_CU[u]) + one_hot_u @ _M_Y_U, rng)

    return DomainSample(x=x, w=w, y=y, z=np.full(n, z), u=u, c=c)


def sample_task(
    task: int,
    n_source: int,
    n_target: int,
    rng: np.random.Generator,
    a_w: float = 1.0,
) -> tuple[list[DomainSample], DomainSample]:
    """Draw source domains (z = 0, 1, ...) and the target domain (z = number of sources).

    The paper uses ``n_source=3200`` per source and ``n_target=9600`` for training; call again
    with a fresh draw for validation and test sets.
    """
    source_p_u0, target_p_u0 = TASKS[task]
    sources = [sample_domain(n_source, p, z, rng, a_w) for z, p in enumerate(source_p_u0)]
    target = sample_domain(n_target, target_p_u0, len(sources), rng, a_w)
    return sources, target
