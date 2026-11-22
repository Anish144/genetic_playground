"""Check weighted predictions against explicit environment calculations."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from genetic_perturbation_playground.data.transfer import batch_controls
from genetic_perturbation_playground.model.batch_transfer import (
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
)
from genetic_perturbation_playground.model.linear_transfer import FullLinearTransfer, LinearTransfer
from genetic_perturbation_playground.model.proxy_unconfounded import mean_inner
from genetic_perturbation_playground.utils.standardisation import Standardised


@pytest.fixture(params=["per_x", "shared_2nn", "hybrid", "hybrid_inf"])
def neural_adapter(virtual_train, request):
    variant = request.param
    settings = (
        {"linear_reg": np.inf if variant == "hybrid_inf" else 0.01}
        if variant.startswith("hybrid")
        else {}
    )
    return NeuralBridgeTransfer(
        variant="hybrid" if variant == "hybrid_inf" else variant,
        n_steps=2,
        n_features_w=2,
        width_w=4,
        depth=1,
        seed=0,
        reg=0.1,
        **settings,
    ).fit(virtual_train)


def test_neural_mixture_equals_environment_average(neural_adapter, virtual_reference):
    bridge = neural_adapter.bridge_
    x = neural_adapter._onehot(np.array([1, 0, 1]))
    controls = batch_controls(virtual_reference)
    expected = 0.25 * bridge.predict(x, controls[0]) + 0.75 * bridge.predict(x, controls[1])
    actual = bridge.predict_mixture(x, controls, np.array([0.25, 0.75]))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    # Independent calculation from network outputs and fitted coefficients.
    with torch.no_grad():
        w = [torch.tensor(c) for c in controls]
        if neural_adapter.variant == "per_x":
            z = 0.25 * bridge.psi_(w[0]).mean(0) + 0.75 * bridge.psi_(w[1]).mean(0)
            manual = z @ bridge.A_
        else:
            z_shared, z_deviation = [
                0.25 * net(w[0]).mean(0) + 0.75 * net(w[1]).mean(0)
                for net in [bridge.shared_network_, bridge.deviation_network_]
            ]
            manual = (
                bridge.intercept_
                + z_shared @ bridge.shared_coef_
                + z_deviation @ bridge.deviation_coef_
            )
            if neural_adapter.variant == "hybrid" and np.isfinite(bridge.linear_reg):
                manual += (0.25 * w[0].mean(0) + 0.75 * w[1].mean(0)) * bridge.linear_coef_
    np.testing.assert_allclose(actual, manual[[1, 0, 1]].numpy(), rtol=1e-6, atol=1e-7)
    assert actual.shape == (3, 2)
    np.testing.assert_array_equal(actual[0], actual[2])


def test_neural_mixture_single_environment_matches_predict(neural_adapter, virtual_reference):
    bridge = neural_adapter.bridge_
    controls = virtual_reference.X
    x = np.eye(2)
    np.testing.assert_allclose(
        bridge.predict_mixture(x, [controls], np.array([1.0])), bridge.predict(x, controls)
    )


def test_neural_mixture_uses_mean_network_outputs(virtual_train):
    bridge = NeuralBridgeTransfer(n_steps=0, n_features_w=2).fit(virtual_train).bridge_
    bridge.psi_ = torch.nn.Sigmoid()
    bridge.A_.fill_(1)
    actual = bridge.predict_mixture(
        np.eye(2), [np.array([[0.0, 0.0], [4.0, 4.0]])], np.array([1.0])
    )
    np.testing.assert_allclose(actual, np.full((2, 2), 0.5 + 1 / (1 + np.exp(-4))))
    assert not np.allclose(actual, 2 / (1 + np.exp(-2)))


def test_neural_mixture_applies_coefficients_once(neural_adapter, virtual_reference, monkeypatch):
    bridge = neural_adapter.bridge_
    original = bridge._predict_embedded
    calls = []
    snapshots = {k: v.clone() for k, v in vars(bridge).items() if isinstance(v, torch.Tensor)}

    def record(embedding, control_mean):
        calls.append((embedding.requires_grad, control_mean.requires_grad, torch.is_grad_enabled()))
        return original(embedding, control_mean)

    monkeypatch.setattr(bridge, "_predict_embedded", record)
    bridge.predict_mixture(np.eye(2), batch_controls(virtual_reference), np.array([0.25, 0.75]))
    assert calls == [(False, False, False)]
    for key, value in snapshots.items():
        torch.testing.assert_close(getattr(bridge, key), value)


def test_neural_mixture_ignores_zero_weight_contribution(neural_adapter, virtual_reference):
    bridge = neural_adapter.bridge_
    controls = batch_controls(virtual_reference)
    np.testing.assert_allclose(
        bridge.predict_mixture(np.eye(2), controls, np.array([0.0, 1.0])),
        bridge.predict(np.eye(2), controls[1]),
    )


@pytest.mark.parametrize(
    "controls, weights",
    [
        ([], []),
        ([np.empty((0, 2))], [1]),
        ([np.ones((2, 2)), np.ones((2, 3))], [0.5, 0.5]),
        ([np.full((1, 2), np.nan)], [1]),
        ([np.full((1, 2), np.inf)], [1]),
        ([np.ones((2, 2))], []),
        ([np.ones((2, 2))], [[1]]),
        ([np.ones((2, 2))], [0]),
        ([np.ones((2, 2))], [2]),
        ([np.ones((2, 2))], [np.nan]),
        ([np.ones((2, 2))], [np.inf]),
        ([np.ones((2, 2)), np.ones((2, 2))], [-1, 2]),
    ],
)
def test_neural_mixture_rejects_invalid_inputs(virtual_train, controls, weights):
    bridge = NeuralBridgeTransfer(n_steps=0, n_features_w=2).fit(virtual_train).bridge_
    with pytest.raises(ValueError):
        bridge.predict_mixture(np.eye(2), controls, np.array(weights))


def _single_environment(model, controls, positions):
    """Use existing fitted coefficients, independent of the new mixture method."""
    raw = model
    offset = 0
    if isinstance(model, Standardised):
        controls = (controls - model.w_mean_) / model.w_std_
        raw, offset = model.model, model.y_mean_
    if isinstance(raw, FullLinearTransfer):
        pred = raw.gamma_[positions] + controls.mean(0) @ raw.slope_.T
    elif isinstance(raw, LinearTransfer):
        pred = raw.gamma_[positions] + raw.slope_ * controls.mean(0)
    elif isinstance(raw, KernelBridgeTransfer):
        inner = mean_inner(raw.kernel_w_, raw.ctrl_, [controls])[:, 0]
        pred = np.concatenate(
            [raw.bridges_[p].predict_embedded(np.zeros(1), inner) for p in positions]
        )
    else:
        pred = raw.bridge_.predict(raw._onehot(np.array(positions)), controls)
    return pred + offset


def test_predict_environment_average_matches_explicit_environment_average(
    transfer_model, virtual_reference
):
    expected = 0.25 * _single_environment(transfer_model, virtual_reference.X[:2], [1, 0, 1])
    expected += 0.75 * _single_environment(transfer_model, virtual_reference.X[2:], [1, 0, 1])
    actual = transfer_model.predict_environment_average(
        ["b", "a", "b"], virtual_reference, environment_weights={"e1": 0.75, "e0": 0.25}
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    if isinstance(transfer_model, LinearTransfer) and transfer_model.slope == 1:
        np.testing.assert_allclose(actual, [[34, 47.5], [14, 27.5], [34, 47.5]])


def test_predict_environment_average_is_invariant_to_control_replication(
    transfer_model, virtual_reference
):
    repeated = ad.concat([virtual_reference, virtual_reference[:2]], index_unique="-")
    expected = transfer_model.predict_environment_average(["a"], virtual_reference)
    np.testing.assert_allclose(
        transfer_model.predict_environment_average(["a"], repeated), expected, rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("labels", [[], ["unknown"]])
def test_predict_environment_average_rejects_unknown_or_empty_requests(
    transfer_model, virtual_reference, labels
):
    with pytest.raises(ValueError):
        transfer_model.predict_environment_average(labels, virtual_reference)


@pytest.mark.parametrize(
    "invalid", ["permutation", "duplicate", "missing_batch", "missing_perturbation", "treated"]
)
def test_predict_environment_average_checks_features_before_standardisation(
    transfer_model, virtual_reference, invalid
):
    reference = virtual_reference.copy()
    if invalid == "permutation":
        reference = reference[:, ::-1].copy()
    elif invalid == "duplicate":
        reference.var_names = ["g0", "g0"]
    elif invalid.startswith("missing_"):
        del reference.obs[invalid[8:]]
    else:
        reference.obs.loc["r0", "perturbation"] = "a"
    with pytest.raises(ValueError):
        transfer_model.predict_environment_average(["a"], reference)


@pytest.mark.parametrize("storage", [sparse.csr_matrix, sparse.csc_matrix])
def test_predict_environment_average_accepts_sparse_reference(
    transfer_model, virtual_reference, storage
):
    expected = transfer_model.predict_environment_average(["b", "a"], virtual_reference)
    virtual_reference.X = storage(virtual_reference.X)
    np.testing.assert_allclose(
        transfer_model.predict_environment_average(["b", "a"], virtual_reference), expected
    )


def test_standardised_preserves_training_feature_names(virtual_train, virtual_reference):
    before = virtual_reference.copy()
    training_names = virtual_train.var_names.copy()
    model = Standardised(LinearTransfer(slope=1)).fit(virtual_train)
    np.testing.assert_array_equal(model.feature_names_in_, ["g0", "g1"])
    pd.testing.assert_frame_equal(model._standardise(virtual_reference).var, virtual_reference.var)
    model.predict_environment_average(["a"], virtual_reference)
    np.testing.assert_array_equal(virtual_reference.X, before.X)
    pd.testing.assert_frame_equal(virtual_reference.obs, before.obs)
    pd.testing.assert_frame_equal(virtual_reference.var, before.var)
    pd.testing.assert_index_equal(virtual_train.var_names, training_names)
    virtual_train.var_names = ["changed", "names"]
    np.testing.assert_array_equal(model.feature_names_in_, ["g0", "g1"])


def test_feature_names_are_available_during_neural_checkpoints(virtual_train, virtual_reference):
    model = Standardised(NeuralBridgeTransfer(n_steps=1, n_features_w=2, width_w=4))
    results = []

    def record(step):
        results.append(model.predict_environment_average(["a"], virtual_reference))

    model.fit(virtual_train, checkpoints=[0, 1], on_checkpoint=record)
    assert len(results) == 2
    assert all(np.isfinite(p).all() and p.shape == (1, 2) for p in results)


def test_new_prediction_rejects_duplicate_training_feature_names(virtual_train, virtual_reference):
    virtual_train.var_names = ["g0", "g0"]
    model = Standardised(LinearTransfer()).fit(virtual_train)
    with pytest.raises(ValueError, match="feature"):
        model.predict_environment_average(["a"], virtual_reference)


def test_linear_environment_average_accumulates_float32_controls_in_float64(virtual_train):
    model = LinearTransfer(slope=1, standardise_w=False).fit(virtual_train)
    values = np.vstack(
        [np.full((10000, 2), 0.1, dtype=np.float32), np.zeros((1, 2), dtype=np.float32)]
    )
    reference = ad.AnnData(
        values,
        obs=pd.DataFrame(
            {"perturbation": ["control"] * 10001, "batch": ["e0"] * 10000 + ["e1"]},
            index=list(map(str, range(10001))),
        ),
        var=virtual_train.var.copy(),
    )
    expected = np.array([[10.0, 20.0]]) + np.float64(np.float32(0.1)) / 2
    predicted = model.predict_environment_average(["a"], reference)
    duplicated = reference[np.r_[np.arange(10000), np.arange(10001)]]
    repeated = model.predict_environment_average(["a"], duplicated)
    np.testing.assert_allclose(predicted, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(repeated, expected, atol=1e-12, rtol=0)


def test_saved_neural_bridge_predicts_without_feature_width_metadata(
    neural_adapter, virtual_reference
):
    import pickle

    bridge = neural_adapter.bridge_
    expected = bridge.predict(np.eye(2), virtual_reference.X)
    del bridge.n_features_in_
    restored = pickle.loads(pickle.dumps(bridge))
    np.testing.assert_array_equal(restored.predict(np.eye(2), virtual_reference.X), expected)
    assert not hasattr(restored, "n_features_in_")  # Prediction must not mutate the saved model.
