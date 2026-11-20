"""Check prediction inputs without using held-out response values."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from genetic_perturbation_playground.data import transfer
from genetic_perturbation_playground.data.transfer import batch_weights, matched_control_mean


@pytest.fixture
def test_cells():
    obs = pd.DataFrame(
        {
            "perturbation": ["control", "a", "c", "a", "control", "a", "c", "control"],
            "batch": ["b1", "b1", "b1", "b0", "b0", "b0", "b0", "b2"],
        },
        index=[str(i) for i in range(8)],
    )
    values = np.full((8, 2), np.nan)
    values[[0, 4, 7]] = [[10, 20], [2, 4], [90, 80]]
    return ad.AnnData(values, obs=obs)


def test_batch_weights_need_only_labels_and_include_control_only_batches(test_cells):
    test_cells.X = None
    np.testing.assert_allclose(
        batch_weights(test_cells, ["c", "a", "c"]),
        [[1 / 2, 1 / 2, 0], [2 / 3, 1 / 3, 0], [1 / 2, 1 / 2, 0]],
    )


@pytest.mark.parametrize("sparse_input", [False, True])
def test_matched_controls_follow_response_counts_with_unavailable_targets(test_cells, sparse_input):
    if sparse_input:
        test_cells.X = sparse.csr_matrix(test_cells.X)
    np.testing.assert_allclose(
        matched_control_mean(test_cells, ["c", "a", "c"]),
        [[6, 12], [14 / 3, 28 / 3], [6, 12]],
    )


@pytest.mark.parametrize("prepare", [batch_weights, matched_control_mean])
def test_prediction_inputs_require_controls_in_each_response_batch(test_cells, prepare):
    without_controls = test_cells[test_cells.obs.index != "4"]
    with pytest.raises(ValueError, match="Every batch with perturbed cells needs control cells"):
        prepare(without_controls, ["a"])


@pytest.mark.parametrize("label", ["b", "z"])
@pytest.mark.parametrize("prepare", [batch_weights, matched_control_mean])
def test_prediction_inputs_reject_missing_perturbations(test_cells, label, prepare):
    with pytest.raises(ValueError, match="Unknown labels"):
        prepare(test_cells, [label])


def test_reference_weights_aligns_named_probabilities(virtual_reference):
    p = transfer.reference_weights(
        virtual_reference,
        feature_names=np.array(["g0", "g1"]),
        environment_weights={"e1": 0.75, "e0": 0.25},
    )
    expected = pd.Series(
        [0.25, 0.75], index=pd.Index(["e0", "e1"], name="batch"), name="probability"
    )
    pd.testing.assert_series_equal(p, expected)


def test_reference_weights_defaults_to_uniform_environments(virtual_reference):
    p = transfer.reference_weights(virtual_reference, feature_names=virtual_reference.var_names)
    np.testing.assert_array_equal(p, [0.5, 0.5])


@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_reference_weights_preserves_sparse_values(virtual_reference, storage):
    virtual_reference.X = storage(virtual_reference.X)
    p = transfer.reference_weights(virtual_reference, feature_names=virtual_reference.var_names)
    np.testing.assert_array_equal(p, [0.5, 0.5])
    np.testing.assert_array_equal(transfer.as_dense(virtual_reference.X), [[0, 2], [2, 4], [5, 9]])


@pytest.mark.parametrize(
    "weights",
    [
        {"e0": 0, "e1": 0},
        {"e0": 1, "e1": 1},
        {"e0": 0.5, "e1": 0.5 + 2e-8},
        {"e0": -1, "e1": 2},
        {"e0": np.nan, "e1": 1},
        {"e0": np.inf, "e1": 0},
        {"e0": 1},
        {"e0": 0.5, "wrong": 0.5},
        {"e0": 0.5, "e1": 0.5, "extra": 0},
    ],
)
def test_invalid_reference_weights_is_rejected(virtual_reference, weights):
    with pytest.raises(ValueError, match="environment_weights"):
        transfer.reference_weights(
            virtual_reference,
            feature_names=virtual_reference.var_names,
            environment_weights=weights,
        )


@pytest.mark.parametrize("weights", [{"e0": 0, "e1": 1}, {"e0": 0.5, "e1": 0.5 + 5e-9}])
def test_reference_weights_accepts_zero_and_normalizes_roundoff(virtual_reference, weights):
    p = transfer.reference_weights(
        virtual_reference, feature_names=virtual_reference.var_names, environment_weights=weights
    )
    expected = np.array(list(weights.values()))
    np.testing.assert_array_equal(p, expected / expected.sum())


@pytest.mark.parametrize(
    "invalid",
    [
        "empty",
        "no_x",
        "no_batch",
        "no_perturbation",
        "null_batch",
        "null_perturbation",
        "treated",
        "nan",
        "inf",
        "sparse_nan",
        "reordered",
        "missing",
        "extra",
        "duplicate",
        "no_features",
    ],
)
def test_invalid_control_reference_is_rejected(virtual_reference, invalid):
    reference = virtual_reference.copy()
    if invalid == "empty":
        reference = reference[:0].copy()
    elif invalid == "no_x":
        reference.X = None
    elif invalid.startswith("no_") and invalid != "no_features":
        del reference.obs[invalid[3:]]
    elif invalid.startswith("null_"):
        reference.obs.loc["r0", invalid[5:]] = None
    elif invalid == "treated":
        reference.obs.loc["r0", "perturbation"] = "a"
    elif invalid in ("nan", "inf", "sparse_nan"):
        reference.X[0, 0] = np.inf if invalid == "inf" else np.nan
        if invalid == "sparse_nan":
            reference.X = sparse.csr_matrix(reference.X)
    elif invalid == "reordered":
        reference = reference[:, ::-1].copy()
    elif invalid == "missing":
        reference = reference[:, :1].copy()
    elif invalid == "extra":
        reference = reference[:, [0, 1, 1]].copy()
        reference.var_names = ["g0", "g1", "g2"]
    elif invalid == "duplicate":
        reference.var_names = ["g0", "g0"]
    elif invalid == "no_features":
        reference = reference[:, :0].copy()
    with pytest.raises(ValueError):
        transfer.reference_weights(reference, feature_names=np.array(["g0", "g1"]))


def test_validate_features_rejects_duplicate_fitted_names(virtual_reference):
    with pytest.raises(ValueError, match="feature"):
        transfer.validate_features(virtual_reference, np.array(["g0", "g0"]))


@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_observed_batch_means_supports_response_only_sparse_data(virtual_train, storage):
    observed = virtual_train[virtual_train.obs.perturbation.ne("control")].copy()
    observed.X = storage(observed.X)
    result = transfer.observed_batch_means(observed, ["b", "a"], "b0")
    assert result.index.tolist() == ["b", "a"]
    assert result.columns.tolist() == ["g0", "g1"]
    np.testing.assert_allclose(result, [[31, 43], [11, 23]])
    assert transfer.observed_batch_means(observed, ["a"], "absent").shape == (0, 2)


def test_observed_batch_means_ignores_unused_categories(virtual_train):
    observed = virtual_train.copy()
    observed.obs["perturbation"] = pd.Categorical(
        observed.obs.perturbation, categories=["control", "a", "b", "unused"]
    )
    observed.obs["batch"] = pd.Categorical(
        observed.obs.batch, categories=["b0", "b1", "b2", "absent"]
    )
    ignored = observed.obs.perturbation.ne("a") | observed.obs.batch.ne("b0")
    observed.X[ignored] = np.nan
    result = transfer.observed_batch_means(observed, ["a", "unused"], "b0")
    assert result.index.tolist() == ["a"]
    np.testing.assert_allclose(result, [[11, 23]])
    assert transfer.observed_batch_means(observed, ["a"], "absent").shape == (0, 2)
