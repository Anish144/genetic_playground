import numpy as np

from genetic_perturbation_playground.utils.kernels import RBF, median_heuristic


def test_rbf_preserves_distances_with_large_offsets_and_per_feature_scales():
    x = np.array([[1e9, -1e9], [1e9 + 3, -1e9 + 4]])
    kernel = RBF(lengthscales=[3, 4])
    expected = np.array([[1, np.exp(-1)], [np.exp(-1), 1]])
    np.testing.assert_allclose(kernel(x), expected)
    np.testing.assert_allclose(kernel(x[:1], x[1:]), expected[:1, 1:])


def test_median_heuristic_matches_brute_force():
    X = np.random.default_rng(0).normal(size=(300, 2))
    dists = [np.linalg.norm(X[i] - X[j]) for i in range(300) for j in range(i + 1, 300)]
    assert np.isclose(median_heuristic(X), np.median(dists))


def test_median_heuristic_subsample_covers_stacked_groups():
    """Data stacked by group: an evenly spaced subsample must not see only the first group."""
    X = np.concatenate([np.zeros(5000), np.full(5000, 10.0)])
    assert median_heuristic(X, max_points=100) > 0  # taking the first 100 rows would give 0
