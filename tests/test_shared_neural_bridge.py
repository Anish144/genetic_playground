"""Check the compact shared/hybrid solver against an independent full-design ridge fit."""

import numpy as np
import pytest
import torch
from neural_reference import pair_data

from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralHybridEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    NeuralSharedEmbeddingBridge,
)
from genetic_perturbation_playground.utils.ridge import NeuralRidgeSolver


def _environments():
    """Unequal counts, duplicate responses, missing categories, and one empty environment."""
    rng = np.random.default_rng(819)
    categories = [[0, 1, 1, 2], [1, 2], [], [1, 2], [1, 2]]
    xs = [np.eye(4)[indices] for indices in categories]  # category 3 is never observed
    ys = [rng.normal(size=(len(indices), 5)) for indices in categories]
    ws = [rng.normal(size=(7 + e, 5)) for e in range(len(categories))]
    counts = [rng.integers(1, 7, size=len(indices)) for indices in categories]
    return xs, ys, ws, counts


def _model(linear_reg=None, **kwargs):
    settings = dict(reg=0.003, deviation_reg=0.007, n_features_w=3, width_w=6, depth=1, seed=1)
    settings.update(kwargs)
    if linear_reg is None:
        return NeuralSharedEmbeddingBridge(**settings)
    return NeuralHybridEmbeddingBridge(linear_reg=linear_reg, **settings)


@pytest.mark.parametrize(
    "model_type",
    [
        NeuralPerXEmbeddingBridge,
        NeuralSharedEmbeddingBridge,
        NeuralHybridEmbeddingBridge,
    ],
)
def test_all_neural_variants_use_the_neural_ridge_solver(model_type):
    xs, ys, ws, counts = _environments()
    model = model_type(n_steps=0, n_features_w=3).fit(xs, ys, ws, counts)
    data = model._to_tensors(xs, ys, ws, counts)
    assert type(model._ridge_solver(data, model._embed(data.w_envs))) is NeuralRidgeSolver


@pytest.mark.parametrize(
    "model_type",
    [NeuralPerXEmbeddingBridge, NeuralSharedEmbeddingBridge, NeuralHybridEmbeddingBridge],
)
def test_fitted_models_with_the_same_settings_are_distinct(model_type):
    xs, ys, ws, counts = _environments()
    original = model_type(n_steps=0, n_features_w=3).fit(xs, ys, ws, counts)
    shifted = model_type(n_steps=0, n_features_w=3).fit(xs, [y + 10 for y in ys], ws, counts)
    assert not np.allclose(original.predict(xs[0], ws[0]), shifted.predict(xs[0], ws[0]))
    assert original != shifted
    assert original == original


def _explicit_ridge(model, pairs, embeddings, fit, scored):
    """Build [category intercept/deviation blocks, shared features, raw coordinate].

    Solve a complete normal equation independently for every output. Work with the
    original rows so duplicate responses contribute their own squared errors.
    """
    shared, deviation = embeddings.chunk(2, dim=1)
    local = torch.cat((torch.ones_like(deviation[:, :1]), deviation), dim=1)
    category_features = (pairs.x[:, :, None] * local[pairs.env, None, :]).flatten(1)
    design = torch.cat((category_features, shared[pairs.env]), dim=1)
    local_penalty = embeddings.new_tensor([model.reg] + [model.deviation_reg] * model.n_features_w)
    penalties = torch.cat(
        (
            local_penalty.repeat(pairs.x.shape[1]),
            embeddings.new_full((model.n_features_w,), model.reg),
        )
    )
    linear_reg = getattr(model, "linear_reg", float("inf"))
    has_linear = np.isfinite(linear_reg)
    if has_linear:
        raw_mean = torch.stack([w.mean(dim=0) for w in pairs.w_envs])
        penalties = torch.cat((penalties, embeddings.new_tensor([linear_reg])))
    weights = pairs.c * fit[pairs.env]
    coefficients, predictions = [], []
    for gene in range(pairs.y.shape[1]):
        features = design
        if has_linear:
            features = torch.cat((features, raw_mean[pairs.env, gene, None]), dim=1)
        gram = features.T @ (weights[:, None] * features)
        gram = gram + weights.sum() * torch.diag(penalties)
        rhs = features.T @ (weights * pairs.y[:, gene])
        coefficient = torch.linalg.solve(gram, rhs)
        coefficients.append(coefficient)
        predictions.append(features @ coefficient)
    coefficients = torch.stack(coefficients, dim=1)
    residual = torch.stack(predictions, dim=1) - pairs.y
    sse = (pairs.c * scored[pairs.env] * residual.square().mean(dim=1)).sum()
    local_size = pairs.x.shape[1] * local.shape[1]
    local_coef = coefficients[:local_size].reshape(pairs.x.shape[1], local.shape[1], -1)
    shared_coef = coefficients[local_size : local_size + model.n_features_w]
    linear_coef = coefficients[-1] if has_linear else embeddings.new_zeros(pairs.y.shape[1])
    return (local_coef, shared_coef, linear_coef), sse


@pytest.mark.parametrize("linear_reg", [None, 0.0, 0.002, float("inf")])
@pytest.mark.parametrize("weighted", [False, True])
def test_solver_coefficients_loss_and_embedding_gradients_match_full_design(linear_reg, weighted):
    xs, ys, ws, counts = _environments()
    counts = counts if weighted else None
    model = _model(linear_reg)
    data = model._to_tensors(xs, ys, ws, counts)
    pairs = pair_data(xs, ys, ws, counts, device=model.device)
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(5, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    fit = torch.tensor([False, True, True, False, True])
    solver = model._ridge_solver(data, embeddings, fit)
    expected_coefs, expected_sse = _explicit_ridge(model, pairs, embeddings, fit, ~fit)
    for actual, expected in zip(solver.coefficients(), expected_coefs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    actual_sse = solver.prediction_sse(~fit)
    torch.testing.assert_close(actual_sse, expected_sse, rtol=1e-8, atol=1e-10)
    actual_grad = torch.autograd.grad(actual_sse, embeddings, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_sse, embeddings)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("linear_reg", [None, 0.002, float("inf")])
@pytest.mark.parametrize("fold_count", [0, 2, 5])
def test_training_loss_and_both_network_gradients_match_full_design(
    monkeypatch, linear_reg, fold_count
):
    xs, ys, ws, counts = _environments()
    model = _model(linear_reg, cross_fit_folds=fold_count, n_features_w=6)
    data = model._to_tensors(xs, ys, ws, counts)
    pairs = pair_data(xs, ys, ws, counts, device=model.device)
    torch.manual_seed(model.seed)
    model._build_networks(data)
    embeddings = model._embed(data.w_envs)
    monkeypatch.setattr(model, "_embed", lambda _: embeddings)
    folds = torch.arange(len(ws)) % max(fold_count, 1)
    monkeypatch.setattr(model, "_env_folds", lambda _: folds)
    actual = model._training_loss(data)
    expected = embeddings.new_zeros(())
    for fold in range(max(fold_count, 1)):
        scored = folds == fold
        fit = ~scored if fold_count else scored
        coefs, sse = _explicit_ridge(model, pairs, embeddings, fit, scored)
        expected = expected + sse
        if not fold_count:  # in-sample loss adds the ridge penalty
            local, shared, linear = coefs
            penalty = model.reg * (local[:, 0].square().sum() + shared.square().sum())
            penalty = penalty + model.deviation_reg * local[:, 1:].square().sum()
            linear_reg = getattr(model, "linear_reg", float("inf"))
            if np.isfinite(linear_reg):
                penalty = penalty + linear_reg * linear.square().sum()
            expected = expected + pairs.c.sum() * penalty / pairs.y.shape[1]
    expected = expected / pairs.c.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    parameters = tuple(model.psi_.parameters())
    actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, parameters)
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-6)
    assert any(torch.count_nonzero(g) for g in actual_grad[: len(parameters) // 2])
    assert any(torch.count_nonzero(g) for g in actual_grad[len(parameters) // 2 :])


@pytest.mark.parametrize("linear_reg", [None, 0.002])
@pytest.mark.parametrize("block_size", [1, 3, 256])
def test_blocked_coefficients_and_new_environment_predictions_match_full_design(
    linear_reg, block_size
):
    xs, ys, ws, counts = _environments()
    model = _model(linear_reg, n_steps=2, gene_block_size=block_size).fit(xs, ys, ws, counts)
    data = model._to_tensors(xs, ys, ws, counts)
    pairs = pair_data(xs, ys, ws, counts, device=model.device)
    with torch.no_grad():
        embeddings = model._embed(data.w_envs)
        all_envs = torch.ones(len(ws), dtype=torch.bool)
        expected, _ = _explicit_ridge(model, pairs, embeddings, all_envs, all_envs)
        local, shared, linear = expected
        torch.testing.assert_close(model.intercept_, local[:, 0])
        torch.testing.assert_close(model.deviation_coef_, local[:, 1:])
        torch.testing.assert_close(model.shared_coef_, shared)
        torch.testing.assert_close(model.linear_coef_, linear)
        controls = np.random.default_rng(32).normal(size=(11, 5))
        shared_mean, deviation_mean = model._embed([torch.as_tensor(controls)]).chunk(2, dim=1)
        prediction = local[:, 0] + shared_mean[0] @ shared + deviation_mean[0] @ local[:, 1:]
        prediction = prediction + torch.as_tensor(controls.mean(0)) * linear
    categories = [2, 0, 2, 3]
    np.testing.assert_allclose(
        model.predict(np.eye(4)[categories], controls), prediction[categories]
    )


@pytest.mark.parametrize("fold_count", [0, 2])
def test_infinite_linear_penalty_exactly_recovers_shared_model(fold_count):
    xs, ys, ws, counts = _environments()
    # Disabled linear term does not require matching proxy/output dimensions.
    ws = [w[:, :2] for w in ws]
    shared = _model(n_steps=3, cross_fit_folds=fold_count).fit(xs, ys, ws, counts)
    hybrid = _model(float("inf"), n_steps=3, cross_fit_folds=fold_count).fit(xs, ys, ws, counts)
    np.testing.assert_array_equal(
        shared.predict(np.eye(4), ws[0]), hybrid.predict(np.eye(4), ws[0])
    )
    assert torch.count_nonzero(hybrid.linear_coef_) == 0


@pytest.mark.parametrize("linear_reg", [None, 0.002])
def test_held_out_targets_do_not_change_fit_coefficients(linear_reg):
    xs, ys, ws, counts = _environments()
    model = _model(linear_reg)
    data = model._to_tensors(xs, ys, ws, counts)
    changed = model._to_tensors(xs, [ys[0] + 1000, *ys[1:]], ws, counts)
    embeddings = torch.randn(5, 6, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    fit = torch.tensor([False, True, True, True, True])
    original = model._ridge_solver(data, embeddings, fit).coefficients()
    modified = model._ridge_solver(changed, embeddings, fit).coefficients()
    for before, after in zip(original, modified, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_zero_unregularised_raw_coordinate_has_zero_slope_and_finite_gradients():
    xs, ys, ws, counts = _environments()
    for controls in ws:
        controls[:, 0] = 0
    model = _model(0.0, n_steps=2, cross_fit_folds=2).fit(xs, ys, ws, counts)
    assert model.linear_coef_[0] == 0
    assert np.isfinite(model.predict(np.eye(4), ws[0])).all()
    assert all(torch.isfinite(p.grad).all() for p in model.psi_.parameters())


def test_hybrid_requires_corresponding_proxy_and_target_dimensions():
    xs, ys, ws, counts = _environments()
    with pytest.raises(ValueError, match="matching proxy and target dimensions"):
        _model(0.01).fit(xs, ys, [w[:, :2] for w in ws], counts)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"reg": 0}, "reg must be finite and positive"),
        ({"reg": float("nan")}, "reg must be finite and positive"),
        ({"deviation_reg": float("inf")}, "deviation_reg must be finite and positive"),
        ({"deviation_reg": -1}, "deviation_reg must be finite and positive"),
        ({"linear_reg": -1}, "linear_reg must be nonnegative"),
        ({"linear_reg": float("nan")}, "linear_reg must be nonnegative"),
        ({"gene_block_size": 0}, "gene_block_size must be positive"),
    ],
)
def test_invalid_penalties_and_block_sizes_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        NeuralHybridEmbeddingBridge(**settings)
