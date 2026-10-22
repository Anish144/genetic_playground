import numpy as np

from genetic_perturbation_playground.utils.metrics import control_whitener


def test_whitener_gives_identity_within_batch_covariance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)) @ rng.normal(size=(5, 5))
    batch = np.repeat(["a", "b"], 2000)
    Z = control_whitener(X, batch, eps=0.0)(X - X.mean(0))
    np.testing.assert_allclose(np.cov(Z.T), np.eye(5), atol=0.1)


def test_whitener_ignores_batch_offsets():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    batch = np.repeat(["a", "b"], 200)
    shifted = X + np.where(batch[:, None] == "a", 10.0, 0.0)
    z = rng.normal(size=(5, 3))
    np.testing.assert_allclose(control_whitener(X, batch)(z), control_whitener(shifted, batch)(z))
