"""Check weighted predictions against explicit environment calculations."""

import numpy as np
import pytest
import torch

from genetic_perturbation_playground.data.transfer import batch_controls
from genetic_perturbation_playground.model.batch_transfer import NeuralBridgeTransfer


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
