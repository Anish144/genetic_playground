import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from genetic_perturbation_playground.data.proxy_unconfounded import (
    conditional_mean,
    outcome_mean,
    sample_environment,
    sample_environments,
)
from genetic_perturbation_playground.model.proxy_confounded import ConditionalMeanEmbedding
from genetic_perturbation_playground.model.proxy_unconfounded import (
    MarginalEmbeddingBridge,
    mean_inner,
)
from genetic_perturbation_playground.utils.kernels import RBF, Kernel


class Linear(Kernel):
    """k(a, b) = a b, so the mean embedding of W is just its sample mean."""

    def K(self, X, X2):
        return X @ X2.T


def _fit(envs, kernel_w=None, reg=1e-3):
    return MarginalEmbeddingBridge(RBF(), kernel_w or RBF(), reg).fit(
        [e.x for e in envs], [e.y for e in envs], [e.w for e in envs]
    )


@pytest.fixture
def small_envs():
    rng = np.random.default_rng(0)
    # unequal n and m per environment, and n != m within an environment
    return [
        sample_environment([t], n, m, rng) for t, n, m in [(-1, 30, 50), (0, 40, 20), (1, 25, 35)]
    ]


# --- data ---------------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [[1.3], [1.3, -0.4, 0.8]])
def test_conditional_mean_matches_monte_carlo(theta):
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 7)
    u = np.array(theta) + rng.standard_normal((1, 200_000, len(theta)))
    mc = outcome_mean(x[:, None], u).mean(axis=1)
    np.testing.assert_allclose(conditional_mean(x, np.array(theta)), mc, atol=5e-3)


@pytest.mark.parametrize("k", [1, 3])
def test_latents_share_law_and_x_is_independent(k):
    rng = np.random.default_rng(0)
    env = sample_environment(np.full(k, 1.5), 50_000, 50_000, rng)
    for u in (env.u1, env.u2):
        assert np.allclose(u.mean(0), 1.5, atol=0.02) and np.allclose(u.std(0), 1, atol=0.02)
    assert abs(env.x.mean()) < 0.02
    assert all(abs(np.corrcoef(env.x, env.u1[:, j])[0, 1]) < 0.02 for j in range(k))


@pytest.mark.parametrize("k", [1, 3])
def test_environments_have_requested_sizes(k):
    envs = sample_environments(4, 10, 7, np.random.default_rng(0), dim_u=k)
    assert len(envs) == 4
    for e in envs:
        assert e.x.shape == e.y.shape == (10,) and e.u1.shape == (10, k)
        assert e.w.shape == e.u2.shape == (7, k) and e.theta.shape == (k,)
        assert np.all(np.abs(e.theta) <= 2)


# --- model: algebra -----------------------------------------------------------------------


def test_fit_matches_explicit_kernel_ridge(small_envs):
    """With a linear W-kernel, <mu_e, mu_e'> = mean(w_e) mean(w_e'); rebuild G by hand."""
    bridge = _fit(small_envs, kernel_w=Linear())
    x = np.concatenate([e.x for e in small_envs])
    y = np.concatenate([e.y for e in small_envs])
    w_bar = np.concatenate([np.full(len(e.x), e.w.mean()) for e in small_envs])
    gram = RBF()(x) * np.outer(w_bar, w_bar)
    coef = cho_solve(cho_factor(gram + 1e-3 * len(x) * np.eye(len(x))), y)
    np.testing.assert_allclose(bridge.coef_, coef, rtol=1e-8)


def test_predict_on_source_environment_reproduces_fitted_values(small_envs):
    bridge = _fit(small_envs)
    x = np.concatenate([e.x for e in small_envs])
    env = np.repeat(np.arange(len(small_envs)), [len(e.x) for e in small_envs])
    gram = RBF()(x) * mean_inner(RBF(), bridge.w_envs_)[np.ix_(env, env)]
    for d, e in enumerate(small_envs):
        np.testing.assert_allclose(bridge.predict(e.x, e.w), (gram @ bridge.coef_)[env == d])


def test_predict_is_bridge_function_averaged_over_target_w(small_envs):
    """predict(x, W*) = mean_j h(x, w*_j), with h(x, w) = sum_i c_i k(x_i, x) <mu_env(i), L(., w)>."""
    bridge = _fit(small_envs)
    w_target = np.random.default_rng(1).normal(0.5, 1.0, 40)
    x_new = np.linspace(-2, 2, 9)

    def h(x, w):
        mu_at_w = np.array([RBF()(w_e, [w]).mean() for w_e in bridge.w_envs_])[bridge.env_]
        return bridge.coef_ @ (RBF()(bridge.x_train_, x) * mu_at_w[:, None])

    expected = np.mean([h(x_new, w) for w in w_target], axis=0)
    np.testing.assert_allclose(bridge.predict(x_new, w_target), expected, rtol=1e-10)


def test_predict_is_invariant_to_order_of_target_w(small_envs):
    bridge = _fit(small_envs)
    w_target = np.random.default_rng(2).normal(size=30)
    x_new = np.linspace(-2, 2, 5)
    np.testing.assert_allclose(
        bridge.predict(x_new, w_target), bridge.predict(x_new, w_target[::-1])
    )


# --- model: statistics --------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("theta_star", [-1.0, 0.5, 1.5])
def test_recovers_target_conditional_mean(theta_star, k):
    """Bridge error is small and far below pooled regression, which ignores the environment."""
    rng = np.random.default_rng(0)
    envs = sample_environments(20, 150, 150, rng, dim_u=k)
    theta = np.full(k, theta_star)
    w_target = sample_environment(theta, 0, 300, rng).w
    x_new = np.linspace(-2.5, 2.5, 100)
    truth = conditional_mean(x_new, theta)

    bridge_mse = np.mean((_fit(envs).predict(x_new, w_target) - truth) ** 2)

    x = np.concatenate([e.x for e in envs])
    y = np.concatenate([e.y for e in envs])
    pooled = ConditionalMeanEmbedding(RBF(), 1e-3).fit(x, y).weights(x_new).T @ y
    pooled_mse = np.mean((pooled - truth) ** 2)

    assert bridge_mse < 0.05
    assert bridge_mse < pooled_mse / 3
