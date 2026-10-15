import numpy as np

from genetic_perturbation_playground.utils.kernels import median_heuristic


def test_median_heuristic_matches_brute_force():
    X = np.random.default_rng(0).normal(size=(300, 2))
    dists = [np.linalg.norm(X[i] - X[j]) for i in range(300) for j in range(i + 1, 300)]
    assert np.isclose(median_heuristic(X), np.median(dists))


def test_median_heuristic_subsample_covers_stacked_groups():
    """Data stacked by group: an evenly spaced subsample must not see only the first group."""
    X = np.concatenate([np.zeros(5000), np.full(5000, 10.0)])
    assert median_heuristic(X, max_points=100) > 0  # taking the first 100 rows would give 0
