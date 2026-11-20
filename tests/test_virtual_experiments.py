"""Numerical contracts for the two virtual-experiment estimators."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from genetic_perturbation_playground.model.linear_transfer import LinearTransfer
from genetic_perturbation_playground.model.virtual_experiments import virtual_experiments


@pytest.fixture
def observed():
    obs = pd.DataFrame({"perturbation": ["a", "a"], "batch": ["e0", "e0"]}, index=["y0", "y1"])
    return ad.AnnData(
        np.array([[100.0, 200.0], [104.0, 204.0]]), obs=obs, var=pd.DataFrame(index=["g0", "g1"])
    )


@pytest.fixture
def linear(virtual_train):
    return LinearTransfer(slope=1, standardise_w=False).fit(virtual_train)


def test_virtual_experiments_returns_both_estimators(linear, virtual_reference, observed):
    weights = {"e1": 0.75, "e0": 0.25}
    result = virtual_experiments(
        linear,
        virtual_reference,
        perturbations=["a", "b"],
        environment_weights=weights,
        observed=observed,
    )
    assert result.obs_names.tolist() == ["a", "b"]
    assert result.obs_names.name == "perturbation"
    assert result.var_names.tolist() == ["g0", "g1"]
    np.testing.assert_allclose(result.X, [[14, 27.5], [34, 47.5]])
    np.testing.assert_allclose(result.layers["observed_fill"], [[36.75, 72.25], [34, 47.5]])
    np.testing.assert_allclose(result.obs["observed_weight"], [0.25, 0.0])
    assert result.uns["environment_weights"] == {"e0": 0.25, "e1": 0.75}
    assert weights == {"e1": 0.75, "e0": 0.25}


def test_virtual_experiments_without_observed_returns_only_bridge(linear, virtual_reference):
    result = virtual_experiments(linear, virtual_reference)
    assert result.obs_names.tolist() == linear.perturbations_.tolist()
    np.testing.assert_allclose(result.X, [[13, 26], [33, 46]])
    assert result.uns["environment_weights"] == {"e0": 0.5, "e1": 0.5}
    assert "observed_fill" not in result.layers
    assert "observed_weight" not in result.obs


def test_virtual_experiments_empty_observed_equals_bridge(linear, virtual_reference, observed):
    result = virtual_experiments(linear, virtual_reference, observed=observed[:0])
    np.testing.assert_array_equal(result.X, result.layers["observed_fill"])
    np.testing.assert_array_equal(result.obs["observed_weight"], [0, 0])


def test_virtual_experiments_complete_observations_equal_weighted_means(
    linear, virtual_reference, observed
):
    extra = ad.AnnData(
        np.array([[50.0, 60.0], [1.0, 2.0], [3.0, 4.0]]),
        obs=pd.DataFrame(
            {"perturbation": ["a", "b", "b"], "batch": ["e1", "e0", "e1"]}, index=["y2", "y3", "y4"]
        ),
        var=observed.var.copy(),
    )
    all_observed = ad.concat([observed, extra])
    result = virtual_experiments(
        linear,
        virtual_reference,
        environment_weights={"e0": 0.25, "e1": 0.75},
        observed=all_observed,
    )
    np.testing.assert_allclose(result.layers["observed_fill"], [[63, 95.5], [2.5, 3.5]])
    np.testing.assert_allclose(result.obs["observed_weight"], [1, 1])


def test_virtual_experiments_observed_zeros_are_present(linear, virtual_reference, observed):
    observed.X[:] = 0
    result = virtual_experiments(linear, virtual_reference[:2], observed=observed)
    np.testing.assert_allclose(result.layers["observed_fill"], [[0, 0], [31, 43]])
    np.testing.assert_array_equal(result.obs["observed_weight"], [1, 0])


def test_virtual_experiments_zero_weight_environment(linear, virtual_reference, observed):
    result = virtual_experiments(
        linear, virtual_reference, observed=observed, environment_weights={"e0": 0.0, "e1": 1.0}
    )
    np.testing.assert_allclose(result.X, [[15, 29], [35, 49]])
    np.testing.assert_array_equal(result.layers["observed_fill"], result.X)
    np.testing.assert_array_equal(result.obs["observed_weight"], [0, 0])


@pytest.mark.parametrize("labels", [[], ["a", "a"], ["unknown"]])
def test_virtual_experiments_rejects_invalid_labels_and_features(linear, virtual_reference, labels):
    with pytest.raises(ValueError):
        virtual_experiments(linear, virtual_reference, perturbations=labels)


def test_virtual_experiments_rejects_reordered_observed_features(
    linear, virtual_reference, observed
):
    with pytest.raises(ValueError, match="feature"):
        virtual_experiments(linear, virtual_reference, observed=observed[:, ::-1])


@pytest.mark.parametrize(
    "bad", ["nan", "inf", "null_perturbation", "null_batch", "no_batch", "no_perturbation", "no_x"]
)
@pytest.mark.parametrize("weights", [None, {"e0": 0.0, "e1": 1.0}])
def test_virtual_experiments_rejects_nonfinite_observations(
    linear, virtual_reference, observed, bad, weights
):
    if bad in ("nan", "inf"):
        observed.X[0, 0] = np.nan if bad == "nan" else np.inf
    elif bad.startswith("null_"):
        observed.obs.loc["y0", bad[5:]] = None
    elif bad == "no_x":
        observed.X = None
    else:
        del observed.obs[bad[3:]]
    with pytest.raises(ValueError):
        virtual_experiments(
            linear, virtual_reference, observed=observed, environment_weights=weights
        )


def test_virtual_experiments_uses_reference_controls(linear, virtual_reference, observed):
    expected = virtual_experiments(linear, virtual_reference, observed=observed)
    other_controls = virtual_reference.copy()
    other_controls.X[:] = np.nan
    combined = ad.concat([observed, other_controls])
    result = virtual_experiments(linear, virtual_reference, observed=combined)
    np.testing.assert_array_equal(result.X, expected.X)
    np.testing.assert_array_equal(result.layers["observed_fill"], expected.layers["observed_fill"])


@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_virtual_experiments_matches_explicit_fill(
    transfer_model, virtual_reference, observed, storage
):
    # Only a/e0 is observed. Request b first to catch positional misalignment.
    e0 = transfer_model.predict_environment_average(["b", "a"], virtual_reference[:2])
    e1 = transfer_model.predict_environment_average(["b", "a"], virtual_reference[2:])
    expected_bridge = 0.25 * e0 + 0.75 * e1
    expected_fill = np.vstack([expected_bridge[0], 0.25 * np.array([102, 202]) + 0.75 * e1[1]])
    observed.X = storage(observed.X)
    original_obs, original_var = observed.obs.copy(), observed.var.copy()
    original_x = observed.X.copy()
    reference_copy = virtual_reference.copy()
    result = virtual_experiments(
        transfer_model,
        virtual_reference,
        perturbations=["b", "a"],
        observed=observed,
        environment_weights={"e0": 0.25, "e1": 0.75},
    )
    np.testing.assert_allclose(result.X, expected_bridge, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(result.layers["observed_fill"], expected_fill, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(result.obs["observed_weight"], [0, 0.25])
    pd.testing.assert_frame_equal(result.var, virtual_reference.var)
    pd.testing.assert_frame_equal(observed.obs, original_obs)
    pd.testing.assert_frame_equal(observed.var, original_var)
    if sparse.issparse(observed.X):
        assert (original_x != observed.X).nnz == 0
    else:
        np.testing.assert_array_equal(original_x, observed.X)
    np.testing.assert_array_equal(virtual_reference.X, reference_copy.X)
    pd.testing.assert_frame_equal(virtual_reference.obs, reference_copy.obs)
    result.var.iloc[0, 0] = "changed"
    pd.testing.assert_frame_equal(virtual_reference.var, reference_copy.var)


def test_virtual_experiments_holds_pairs_out_of_both_estimators(virtual_train):
    held = virtual_train.obs.perturbation.eq("a") & virtual_train.obs.batch.eq("b1")
    controls = virtual_train.obs.perturbation.eq("control")
    available = virtual_train[~held]
    model = LinearTransfer().fit(available)
    coefficient = model.coef_.copy()
    before = virtual_experiments(model, virtual_train[controls], observed=available)
    virtual_train.X[held] += 10000
    after = virtual_experiments(model, virtual_train[controls], observed=virtual_train[~held])
    np.testing.assert_array_equal(before.X, after.X)
    np.testing.assert_array_equal(before.layers["observed_fill"], after.layers["observed_fill"])
    np.testing.assert_array_equal(model.coef_, coefficient)
    np.testing.assert_allclose(after.obs["observed_weight"], [2 / 3, 1])


def test_virtual_experiments_round_trips_h5ad(linear, virtual_reference, observed, tmp_path):
    result = virtual_experiments(linear, virtual_reference, observed=observed)
    output = tmp_path / "virtual.h5ad"
    with ad.settings.override(allow_write_nullable_strings=True):
        result.write_h5ad(output)
    loaded = ad.read_h5ad(output)
    np.testing.assert_array_equal(loaded.X, result.X)
    np.testing.assert_array_equal(loaded.layers["observed_fill"], result.layers["observed_fill"])
    pd.testing.assert_frame_equal(loaded.obs, result.obs, check_dtype=False, check_index_type=False)
    pd.testing.assert_frame_equal(loaded.var, result.var, check_dtype=False, check_index_type=False)
    assert loaded.uns["environment_weights"] == result.uns["environment_weights"]


def test_virtual_experiments_corrects_only_observed_pairs(
    linear, virtual_reference, observed, monkeypatch
):
    calls = []
    original = linear.predict_environment_average

    def record(labels, reference, **kwargs):
        calls.append((labels, sorted(reference.obs.batch.unique())))
        return original(labels, reference, **kwargs)

    monkeypatch.setattr(linear, "predict_environment_average", record)
    result = virtual_experiments(linear, virtual_reference, observed=observed)
    assert calls == [(["a", "b"], ["e0", "e1"]), (["a"], ["e0"])]
    np.testing.assert_allclose(result.layers["observed_fill"], [[58.5, 115.5], [33, 46]])
