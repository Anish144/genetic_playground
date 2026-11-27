"""Check shared and hybrid solvers against an explicit ridge fit."""

import numpy as np
import pytest
import torch
from neural_reference import pair_data

from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralHybridEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    NeuralSharedEmbeddingBridge,
)
from genetic_perturbation_playground.utils.ridge import GramRidgeSolver


def _environments(n_outputs=5):
    """Unequal counts, duplicate responses, missing categories, and one empty environment."""
    rng = np.random.default_rng(819)
    categories = [[0, 1, 1, 2], [1, 2], [], [1, 2], [1, 2]]
    xs = [np.eye(4)[indices] for indices in categories]  # category 3 is never observed
    ys = [rng.normal(size=(len(indices), n_outputs)) for indices in categories]
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
    assert type(model._ridge_solver(data, model._embed(data.w_envs))) is GramRidgeSolver


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
    """Fit an explicit design with category and shared feature blocks.

    Include control means in the shared block when the hybrid's linear term is
    enabled. Keep duplicate response rows so each contributes its own error.
    Return ((local, shared), sse, penalty): sse is count-weighted and averaged
    over outputs; penalty sums reg * coefficient**2 before scaling by N / J.
    """
    q = model.n_features_w
    shared, deviation = embeddings[:, :-q], embeddings[:, -q:]
    local = torch.cat((torch.ones_like(deviation[:, :1]), deviation), dim=1)
    category_features = (pairs.x[:, :, None] * local[pairs.env, None, :]).flatten(1)
    design = torch.cat((category_features, shared[pairs.env]), dim=1)
    local_penalty = embeddings.new_tensor([model.reg] + [model.deviation_reg] * q)
    shared_penalty = embeddings.new_full((shared.shape[1],), getattr(model, "linear_reg", 0.0))
    shared_penalty[:q] = model.reg
    penalties = torch.cat((local_penalty.repeat(pairs.x.shape[1]), shared_penalty))
    weights = pairs.c * fit[pairs.env]
    gram = design.T @ (weights[:, None] * design) + weights.sum() * torch.diag(penalties)
    coefficients = torch.linalg.solve(gram, design.T @ (weights[:, None] * pairs.y))
    residual = design @ coefficients - pairs.y
    sse = (pairs.c * scored[pairs.env] * residual.square().mean(dim=1)).sum()
    penalty = (penalties[:, None] * coefficients.square()).sum()
    local_size = pairs.x.shape[1] * local.shape[1]
    local_coef = coefficients[:local_size].reshape(pairs.x.shape[1], local.shape[1], -1)
    return (local_coef, coefficients[local_size:]), sse, penalty


def _width(model, d_w=5):
    """Return 2q plus d_w when the hybrid's linear term is enabled."""
    has_b = np.isfinite(getattr(model, "linear_reg", float("inf")))
    return 2 * model.n_features_w + d_w * has_b


@pytest.mark.parametrize("linear_reg", [None, 0.002, float("inf")])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n_outputs", [3, 8])  # D = 5: direct path, then Gram path
def test_solver_coefficients_loss_and_embedding_gradients_match_full_design(
    linear_reg, weighted, n_outputs
):
    xs, ys, ws, counts = _environments(n_outputs)
    counts = counts if weighted else None
    model = _model(linear_reg)
    data = model._to_tensors(xs, ys, ws, counts)
    pairs = pair_data(xs, ys, ws, counts, device=model.device)
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(
        5, _width(model), generator=generator, dtype=torch.float64, requires_grad=True
    )
    fit = torch.tensor([False, True, True, False, True])
    solver = model._ridge_solver(data, embeddings, fit)
    expected_coefs, expected_sse, _ = _explicit_ridge(model, pairs, embeddings, fit, ~fit)
    for actual, expected in zip(solver.coefficients(), expected_coefs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    actual_sse = solver.prediction_sse(~fit)
    torch.testing.assert_close(actual_sse, expected_sse, rtol=1e-8, atol=1e-10)
    actual_grad = torch.autograd.grad(actual_sse, embeddings, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_sse, embeddings)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("linear_reg", [None, 0.002, float("inf")])
@pytest.mark.parametrize("fold_count", [0, 2, 5])
@pytest.mark.parametrize("n_outputs", [3, 8])  # D = 5: direct path, then Gram path
def test_training_loss_and_both_network_gradients_match_full_design(
    monkeypatch, linear_reg, fold_count, n_outputs
):
    xs, ys, ws, counts = _environments(n_outputs)
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
        _, sse, penalty = _explicit_ridge(model, pairs, embeddings, fit, scored)
        expected = expected + sse
        if not fold_count:  # in-sample loss adds the ridge penalty
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
    q = model.n_features_w
    with torch.no_grad():
        embeddings = model._embed(data.w_envs)
        all_envs = torch.ones(len(ws), dtype=torch.bool)
        (local, shared), _, _ = _explicit_ridge(model, pairs, embeddings, all_envs, all_envs)
        torch.testing.assert_close(model.intercept_, local[:, 0])
        torch.testing.assert_close(model.deviation_coef_, local[:, 1:])
        torch.testing.assert_close(model.shared_coef_, shared)
        controls = np.random.default_rng(32).normal(size=(11, 5))
        embedded = model._embed([torch.as_tensor(controls)])[0]
        prediction = local[:, 0] + embedded[:-q] @ shared + embedded[-q:] @ local[:, 1:]
    categories = [2, 0, 2, 3]
    np.testing.assert_allclose(
        model.predict(np.eye(4)[categories], controls), prediction[categories]
    )


@pytest.mark.parametrize("fold_count", [0, 2])
def test_infinite_linear_penalty_exactly_recovers_shared_model(fold_count):
    xs, ys, ws, counts = _environments()
    shared = _model(n_steps=3, cross_fit_folds=fold_count).fit(xs, ys, ws, counts)
    hybrid = _model(float("inf"), n_steps=3, cross_fit_folds=fold_count).fit(xs, ys, ws, counts)
    np.testing.assert_array_equal(
        shared.predict(np.eye(4), ws[0]), hybrid.predict(np.eye(4), ws[0])
    )
    assert hybrid.full_coef_.shape == (5, 5) and not hybrid.full_coef_.any()


@pytest.mark.parametrize("linear_reg", [None, 0.002])
def test_held_out_targets_do_not_change_fit_coefficients(linear_reg):
    xs, ys, ws, counts = _environments()
    model = _model(linear_reg)
    data = model._to_tensors(xs, ys, ws, counts)
    changed = model._to_tensors(xs, [ys[0] + 1000, *ys[1:]], ws, counts)
    generator = torch.Generator().manual_seed(8)
    embeddings = torch.randn(5, _width(model), generator=generator, dtype=torch.float64)
    fit = torch.tensor([False, True, True, True, True])
    original = model._ridge_solver(data, embeddings, fit).coefficients()
    modified = model._ridge_solver(changed, embeddings, fit).coefficients()
    for before, after in zip(original, modified, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"reg": 0}, "reg must be finite and positive"),
        ({"reg": float("nan")}, "reg must be finite and positive"),
        ({"deviation_reg": float("inf")}, "deviation_reg must be finite and positive"),
        ({"deviation_reg": -1}, "deviation_reg must be finite and positive"),
        ({"linear_reg": 0}, "linear_reg must be positive"),
        ({"linear_reg": -1}, "linear_reg must be positive"),
        ({"linear_reg": float("nan")}, "linear_reg must be positive"),
        ({"gene_block_size": 0}, "gene_block_size must be positive"),
    ],
)
def test_invalid_penalties_and_block_sizes_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        NeuralHybridEmbeddingBridge(**settings)


@pytest.mark.parametrize("cross_fit_folds", [0, 2])
@pytest.mark.parametrize("d_w", [3, 8])  # fewer and more control coordinates than D = 5
def test_hybrid_matches_dense_full_ridge(d_w, cross_fit_folds):
    xs, ys, ws, counts = _environments()
    ws = [np.random.default_rng(e).normal(size=(len(w), d_w)) for e, w in enumerate(ws)]
    model = NeuralHybridEmbeddingBridge(
        reg=0.003,
        deviation_reg=0.007,
        linear_reg=0.02,
        n_features_w=2,
        width_w=6,
        depth=1,
        n_steps=2,
        seed=1,
        cross_fit_folds=cross_fit_folds,
    ).fit(xs, ys, ws, counts)
    data = model._to_tensors(xs, ys, ws, counts)
    pairs = pair_data(xs, ys, ws, counts, device=model.device)
    q, rank = model.n_features_w, min(d_w, len(ws))
    with torch.no_grad():
        embeddings = model._embed(data.w_envs)
        loss = model._training_loss(data)
    means = torch.stack([w.mean(0) for w in pairs.w_envs])
    basis = model.control_basis_
    assert basis.shape == (d_w, rank)
    assert model.shared_coef_.shape == (q + rank, 5)
    # Training means lie in their own row space, so projecting them loses nothing.
    torch.testing.assert_close(embeddings[:, q:-q] @ basis.T, means)

    # Dense ridge on all d_w raw control coordinates, without any projection.
    raw = torch.cat((embeddings[:, :q], means, embeddings[:, -q:]), dim=1)
    every = torch.ones(len(ws), dtype=torch.float64)
    (local_coef, shared_coef), sse, penalty = _explicit_ridge(model, pairs, raw, every, every)
    if not cross_fit_folds:  # the cross-fitted loss is held-out error, not this objective
        N = pairs.c.sum()
        torch.testing.assert_close(loss, (sse + N * penalty / 5) / N, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(model.full_coef_, shared_coef[q:], rtol=1e-6, atol=1e-8)

    def dense_predict(local, shared):  # (G, J) from (q,) deviation and shared features
        return torch.cat((torch.ones(1), local)) @ local_coef + shared @ shared_coef

    np.testing.assert_allclose(
        model.predict(np.eye(4), ws[0]),
        dense_predict(raw[0, -q:], raw[0, :-q]).numpy(),
        rtol=1e-6,
        atol=1e-8,
    )
    # New controls outside the training row space meet B exactly.
    w_new = torch.as_tensor(np.random.default_rng(99).normal(size=(9, d_w)))
    with torch.no_grad():
        deviation = model.deviation_network_(w_new).mean(0)
        shared = torch.cat((model.shared_network_(w_new).mean(0), w_new.mean(0)))
    np.testing.assert_allclose(
        model.predict(np.eye(4), w_new.numpy()),
        dense_predict(deviation, shared).numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_hybrid_zero_controls_reduces_rank():
    xs, ys, ws, counts = _environments()
    ws = [np.zeros((len(w), 4)) for w in ws]
    model = _model(0.01, n_steps=1).fit(xs, ys, ws, counts)
    assert model.control_basis_.shape == (4, 0)
    assert model.full_coef_.shape == (4, 5) and not model.full_coef_.any()
    assert np.isfinite(model.predict(np.eye(4), ws[0])).all()
