"""Check the linear ridge penalty and its zero-penalty limit."""

import numpy as np
import pytest

from genetic_perturbation_playground.utils import ridge


@pytest.fixture
def linear_stats():
    """One category in two environments; the second control coordinate is constant."""
    return {
        "n": np.array([2]),
        "y_mean": np.array([[3.0, 7.0]]),
        "w_mean": np.array([[1.0, 10.0]]),
        "W": np.array([[0.0, 10.0], [2.0, 10.0]]),
        "batch_n": np.array([1, 1]),
        "batch_y_sum": np.array([[1.0, 5.0], [5.0, 9.0]]),
    }


@pytest.mark.parametrize("reg, expected", [(0.0, 2.0), (1.0, 1.0), (3.0, 0.5)])
def test_linear_ridge_shrinks_slope_and_handles_constant_controls(linear_stats, reg, expected):
    with np.errstate(all="raise"):
        slopes = ridge.fit_linear_slopes(linear_stats, reg=reg)
    np.testing.assert_allclose(slopes, [expected, 0.0])


def test_linear_ridge_uses_a_positive_penalty_by_default(linear_stats):
    assert 0 < ridge.fit_linear_slopes(linear_stats)[0] < 2


def test_linear_ridge_is_unchanged_when_all_response_counts_are_repeated(linear_stats):
    repeated = {name: value.copy() for name, value in linear_stats.items()}
    for name in ("n", "batch_n", "batch_y_sum"):
        repeated[name] *= 7
    np.testing.assert_allclose(
        ridge.fit_linear_slopes(linear_stats, reg=0.3),
        ridge.fit_linear_slopes(repeated, reg=0.3),
    )


@pytest.mark.parametrize("reg", [0.0, 1e-3, 1.0])
@pytest.mark.parametrize("control_offset, target_offset", [(1e8, 0.0), (0.0, 1e8), (1e8, 1e8)])
def test_linear_ridge_slopes_ignore_large_offsets(linear_stats, reg, control_offset, target_offset):
    shifted = {name: value.copy() for name, value in linear_stats.items()}
    shifted["W"] += control_offset
    shifted["w_mean"] += control_offset
    shifted["y_mean"] += target_offset
    shifted["batch_y_sum"] += shifted["batch_n"][:, None] * target_offset
    original = {name: value.copy() for name, value in shifted.items()}

    slopes = ridge.fit_linear_slopes(shifted, reg=reg)

    np.testing.assert_allclose(slopes, [2 / (1 + reg), 0.0])
    for name in shifted:
        np.testing.assert_array_equal(shifted[name], original[name])


@pytest.mark.parametrize("reg", [-0.1, np.nan, np.inf, -np.inf])
def test_linear_ridge_rejects_invalid_penalties(linear_stats, reg):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ridge.fit_linear_slopes(linear_stats, reg=reg)


def test_linear_ridge_requires_positive_training_weight(linear_stats):
    linear_stats["n"][:] = 0
    with pytest.raises(ValueError, match="positive total response weight"):
        ridge.fit_linear_slopes(linear_stats)
