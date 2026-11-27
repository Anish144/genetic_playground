"""Check the direct ridge solver against the Gram solver on the same data."""

import numpy as np
import pytest
import torch

from genetic_perturbation_playground.model import proxy_unconfounded_neural
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralHybridEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    NeuralSharedEmbeddingBridge,
)
from genetic_perturbation_playground.utils.ridge import (
    DirectRidgeSolver,
    GramRidgeSolver,
    NeuralRidgeSolver,
)

Q, Q_SHARED, Q_LINEAR = 3, 3, 2


def _environments(n_outputs=5):
    """Five environments: duplicate rows, a never-observed category 3 and an empty environment."""
    rng = np.random.default_rng(819)
    categories = [[0, 1, 1, 2], [1, 2], [], [1, 2], [1, 2]]
    xs = [np.eye(4)[indices] for indices in categories]
    ys = [rng.normal(size=(len(indices), n_outputs)) for indices in categories]
    ws = [rng.normal(size=(7 + e, 5)) for e in range(len(categories))]
    counts = [rng.integers(1, 7, size=len(indices)) for indices in categories]
    return xs, ys, ws, counts


def _with_gram(data):
    """The same data with target_gram built, whatever the shape."""
    return data._replace(target_gram=data.cy @ data.cy.mT / data.cy.shape[-1])


def _width(variant):
    return {"per_x": Q, "shared": Q_SHARED + Q, "hybrid": Q_SHARED + Q_LINEAR + Q}[variant]


def _solver(cls, data, embeddings, fit, variant):
    if variant == "per_x":
        return cls.per_category(data, embeddings, fit, reg=0.003)
    shared_reg = [0.003] * Q_SHARED + ([0.02] * Q_LINEAR if variant == "hybrid" else [])
    return cls.shared_deviation(
        data,
        embeddings,
        fit,
        reg=0.003,
        deviation_reg=0.007,
        shared_reg=torch.tensor(shared_reg, dtype=torch.float64),
    )


@pytest.mark.parametrize("n_outputs, gram", [(8, True), (5, True), (3, False)])  # D = 5
def test_facade_uses_gram_only_when_environments_do_not_exceed_outputs(n_outputs, gram):
    xs, ys, ws, counts = _environments(n_outputs)
    data = NeuralPerXEmbeddingBridge()._to_tensors(xs, ys, ws, counts)
    assert NeuralRidgeSolver.uses_gram(5, n_outputs) is gram
    assert (data.target_gram is not None) is gram
    assert NeuralRidgeSolver.solver_type(data) is (GramRidgeSolver if gram else DirectRidgeSolver)


@pytest.mark.parametrize("variant", ["per_x", "shared", "hybrid"])
@pytest.mark.parametrize("n_outputs", [3, 5, 8])  # D = 5: J < D, J = D, J > D
@pytest.mark.parametrize("fit", [None, [False, True, True, False, True]])
def test_direct_solver_matches_gram_solver(variant, n_outputs, fit):
    xs, ys, ws, counts = _environments(n_outputs)
    data = _with_gram(NeuralPerXEmbeddingBridge()._to_tensors(xs, ys, ws, counts))
    generator = torch.Generator().manual_seed(3)
    emb = torch.randn(5, _width(variant), generator=generator, dtype=torch.float64)
    emb.requires_grad_(True)
    fit = None if fit is None else torch.tensor(fit)
    scored = None if fit is None else ~fit
    gram = _solver(GramRidgeSolver, data, emb, fit, variant)
    direct = _solver(DirectRidgeSolver, data._replace(target_gram=None), emb, fit, variant)
    for actual, expected in zip(direct.coefficients(), gram.coefficients(), strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(direct.penalty(), gram.penalty(), rtol=1e-8, atol=1e-10)
    sse_direct, sse_gram = direct.prediction_sse(scored), gram.prediction_sse(scored)
    torch.testing.assert_close(sse_direct, sse_gram, rtol=1e-8, atol=1e-10)
    grad_direct = torch.autograd.grad(sse_direct + direct.penalty(), emb)[0]
    grad_gram = torch.autograd.grad(sse_gram + gram.penalty(), emb)[0]
    torch.testing.assert_close(grad_direct, grad_gram, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("n_outputs, checkpointed", [(3, True), (8, False)])  # D = 5
@pytest.mark.parametrize("model_type", [NeuralSharedEmbeddingBridge, NeuralHybridEmbeddingBridge])
def test_shared_cross_fit_checkpoints_folds_only_on_the_direct_path(
    monkeypatch, model_type, n_outputs, checkpointed
):
    calls, original = [], proxy_unconfounded_neural.checkpointed_sse

    def recording(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(proxy_unconfounded_neural, "checkpointed_sse", recording)
    xs, ys, ws, counts = _environments(n_outputs)
    model_type(n_steps=1, n_features_w=3, cross_fit_folds=2).fit(xs, ys, ws, counts)
    assert bool(calls) is checkpointed


@pytest.mark.parametrize("cls", [GramRidgeSolver, DirectRidgeSolver])
def test_shared_fit_without_training_weight_raises_value_error(cls):
    xs, ys, ws, counts = _environments(3)
    data = _with_gram(NeuralPerXEmbeddingBridge()._to_tensors(xs, ys, ws, counts))
    emb = torch.randn(5, _width("shared"), dtype=torch.float64)
    no_weight = torch.tensor([False, False, True, False, False])  # environment 2 has no pairs
    with pytest.raises(ValueError, match="positive total response weight"):
        _solver(cls, data, emb, no_weight, "shared")
