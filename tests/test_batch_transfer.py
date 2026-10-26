import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import wandb

from genetic_perturbation_playground.model.batch_transfer import (
    BatchTransferModel,
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
)
from genetic_perturbation_playground.model.proxy_unconfounded import MarginalEmbeddingBridge
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralMarginalEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    _features,
    _mlp,
    _weighted_ridge,
    _weighted_sse,
)
from genetic_perturbation_playground.utils.kernels import RBF


def _cells(rng, n_batches=4, perts=("a", "b", "c"), n_genes=5):
    obs = [("control", f"b{e}") for e in range(n_batches) for _ in range(rng.integers(5, 10))]
    obs += [
        (p, f"b{e}")
        for e in range(n_batches)
        for p in perts
        if rng.random() < 0.8
        for _ in range(rng.integers(1, 6))
    ]
    obs = pd.DataFrame(obs, columns=["perturbation", "batch"])
    return ad.AnnData(rng.normal(size=(len(obs), n_genes)), obs=obs)


def test_matches_cell_level_marginal_embedding_bridge():
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    model = KernelBridgeTransfer(reg=1e-2).fit(train)

    # the same estimator fit on cells: one-hot x with a delta kernel, w = control cells
    perts = model.perturbations_
    xs, ys, ws = [], [], []
    for e in sorted(set(train.obs["batch"])):
        b = train[train.obs["batch"] == e]
        ctrl = b.obs["perturbation"] == "control"
        xs.append(np.eye(len(perts))[np.searchsorted(perts, b.obs["perturbation"][~ctrl])])
        ys.append(b.X[~ctrl.to_numpy()])
        ws.append(b.X[ctrl.to_numpy()])
    query = sorted(set(test.obs["perturbation"]) - {"control"})
    x_query = np.eye(len(perts))[np.searchsorted(perts, query)]
    w_test = test.X[(test.obs["perturbation"] == "control").to_numpy()]
    bridge = MarginalEmbeddingBridge(RBF(lengthscales=1e-2), model.kernel_w_, reg=1e-2)
    expected = np.stack(
        [bridge.fit(xs, [y[:, g] for y in ys], ws).predict(x_query, w_test) for g in range(5)], 1
    )
    np.testing.assert_allclose(model.predict(query, test), expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("cross_fit_folds", [0, 2, 3])
def test_neural_bridge_counts_match_replicated_rows(cross_fit_folds):
    """A pair with count c fits exactly like c identical rows (weighted loss and ridge)."""
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(6, 2)) for _ in range(3)]
    ys = [rng.normal(size=(6, 4)) for _ in range(3)]  # multi-output y
    ws = [rng.normal(size=(10, 3)) for _ in range(3)]
    counts = [rng.integers(1, 4, size=6) for _ in range(3)]

    def fit(xs, ys, counts=None):
        bridge = NeuralMarginalEmbeddingBridge(
            n_steps=5,
            weight_decay=1e-2,
            n_features=4,
            n_features_w=3,
            cross_fit_folds=cross_fit_folds,
        )
        return bridge.fit(xs, ys, ws, counts)

    weighted = fit(xs, ys, counts)
    replicated = fit(
        [np.repeat(x, c, 0) for x, c in zip(xs, counts, strict=True)],
        [np.repeat(y, c, 0) for y, c in zip(ys, counts, strict=True)],
    )
    x_new = rng.normal(size=(5, 2))
    np.testing.assert_allclose(
        weighted.predict(x_new, ws[0]), replicated.predict(x_new, ws[0]), rtol=1e-6
    )


def test_score_truth_uses_only_unseen_controls():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from train_batch_transfer_expression import MODELS, score

    rng = np.random.default_rng(0)
    data = _cells(rng, n_batches=1, perts=("a",))
    is_ctrl = (data.obs["perturbation"] == "control").to_numpy()
    seen = np.arange(data.n_obs) % 2 == 0
    models = {n: BatchTransferModel(s).fit(data) for n, s in MODELS.items()}
    s = score(models, data, 2, seen)

    effect = data.X[~is_ctrl].mean(0) - data.X[is_ctrl & ~seen].mean(0)
    mse = s[s.metric == "mse"].set_index("model")["value"]
    np.testing.assert_allclose(mse["no effect"], np.mean(effect**2))
    # shift predicts gamma + (seen control mean); the shared baseline cancels in its error
    pred = models["shift (b=1)"].gamma_[0] + data.X[is_ctrl & seen].mean(0)
    np.testing.assert_allclose(mse["shift (b=1)"], np.mean((pred - data.X[~is_ctrl].mean(0)) ** 2))


def test_unweighted_bridge_ignores_cell_counts():
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    # duplicating batch b0's perturbed cells doubles their counts but leaves every pair mean as is
    extra = train[
        (train.obs["batch"] == "b0").to_numpy() & (train.obs["perturbation"] != "control")
    ]
    doubled = ad.concat([train, extra])
    perts = sorted(set(test.obs["perturbation"]) - {"control"})

    def pred(d, weighted):
        return KernelBridgeTransfer(1e-2, weighted).fit(d).predict(perts, test)

    np.testing.assert_allclose(pred(train, False), pred(doubled, False))
    assert not np.allclose(pred(train, True), pred(doubled, True))


def test_neural_checkpoint_equals_fit_with_that_many_steps():
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    perts = sorted(set(test.obs["perturbation"]) - {"control"})
    kwargs = {"n_features": 4, "n_features_w": 3, "n_steps": 6}
    at_step = {}

    model = NeuralBridgeTransfer(**kwargs)
    model.fit(train, [3], lambda step: at_step.update({step: model.predict(perts, test)}))

    short = NeuralBridgeTransfer(**{**kwargs, "n_steps": 3}).fit(train)
    np.testing.assert_allclose(at_step[3], short.predict(perts, test))


def test_neural_bridge_logs_loss_every_step_only_with_a_log_key(monkeypatch):
    rng = np.random.default_rng(0)
    logged, axes = [], []
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", logged.append)
    monkeypatch.setattr(wandb, "define_metric", lambda key, step_metric: axes.append(step_metric))
    xs, ys, ws = ([rng.normal(size=(5, 2)) for _ in range(2)] for _ in range(3))
    NeuralMarginalEmbeddingBridge(n_steps=3).fit(xs, ys, ws)
    assert logged == axes == []
    NeuralMarginalEmbeddingBridge(n_steps=3, log_key="fold 2/neural bridge").fit(xs, ys, ws)
    assert axes == ["fold 2/neural bridge/step"]
    assert [m["fold 2/neural bridge/step"] for m in logged] == [0, 1, 2]
    assert all(np.isfinite(m["fold 2/neural bridge/loss"]) for m in logged)


def test_cross_fit_needs_two_environments():
    rng = np.random.default_rng(0)
    bridge = NeuralMarginalEmbeddingBridge(n_steps=1, cross_fit_folds=2)
    with pytest.raises(ValueError, match="two environments"):
        bridge.fit([rng.normal(size=5)], [rng.normal(size=5)], [rng.normal(size=(4, 2))])


def test_cross_fit_with_one_fold_per_environment_is_leave_one_environment_out():
    """K = D: each environment's error under A solved on all the others, pooled by weight."""
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(6, 2)) for _ in range(4)]
    ys = [rng.normal(size=(6, 3)) for _ in range(4)]
    ws = [rng.normal(size=(10, 2)) for _ in range(4)]
    counts = [rng.integers(1, 4, size=6) for _ in range(4)]
    bridge = NeuralMarginalEmbeddingBridge(n_steps=2, n_features=3, n_features_w=2)
    bridge.fit(xs, ys, ws, counts)
    bridge.cross_fit_folds = 4
    data = bridge._to_tensors(xs, ys, ws, counts)

    with torch.no_grad():
        loss = bridge._cross_fit_loss(data)
        feats = _features(bridge.phi_(data.x), bridge._embed(data.w_envs)[data.env])
        sse = 0
        for e in range(4):
            held = data.env == e
            A = _weighted_ridge(feats[~held], data.y[~held], data.c[~held], bridge.reg)
            sse += _weighted_sse(feats[held] @ A, data.y[held], data.c[held])
    torch.testing.assert_close(loss, sse / data.c.sum())


def test_neural_bridge_sizes_phi_and_psi_separately():
    rng = np.random.default_rng(0)
    xs, ys, ws = ([rng.normal(size=(5, 2)) for _ in range(2)] for _ in range(3))
    bridge = NeuralMarginalEmbeddingBridge(
        n_steps=1, width_x=3, n_features=4, width_w=5, n_features_w=6
    ).fit(xs, ys, ws)

    def last_two_sizes(mlp):
        return [m.out_features for m in mlp.modules() if isinstance(m, torch.nn.Linear)][-2:]

    assert last_two_sizes(bridge.phi_) == [3, 4]  # hidden width, output dimension
    assert last_two_sizes(bridge.psi_) == [5, 6]


def test_mlp_hidden_layers_are_swiglu_with_residual_connections_where_sizes_match():
    torch.manual_seed(0)
    mlp = _mlp(d_in=3, width=4, depth=2, d_out=5)
    first, second, out = mlp
    x = torch.randn(6, 3, dtype=torch.float64)

    def swiglu(layer, h):
        return torch.nn.functional.silu(layer.gate(h)) * layer.value(h)

    h1 = swiglu(first, x)  # 3 -> 4 units: no residual
    h2 = h1 + swiglu(second.layer, h1)  # 4 -> 4 units: residual
    torch.testing.assert_close(mlp(x), out(h2))


def _categorical_envs(rng, n_envs=4, n_cats=3, d_y=2):
    """One-hot xs (every category in every environment, but the last misses category 0), ys, ws
    and counts."""
    cats = [np.arange(n_cats)[e == n_envs - 1 :] for e in range(n_envs)]
    xs = [np.eye(n_cats)[c] for c in cats]
    ys = [rng.normal(size=(len(c), d_y)) for c in cats]
    ws = [rng.normal(size=(7, 2)) for _ in range(n_envs)]
    counts = [rng.integers(1, 4, size=len(c)) for c in cats]
    return xs, ys, ws, counts


def _dense_fit(bridge, xs, ys, ws, counts, held=None):
    """The shared bridge's fit with phi(x) = x on `bridge`'s psi (features x ⊗ mu_e): A solved
    without environment `held` and its weighted sse on `held` (on all pairs if None)."""
    data = NeuralMarginalEmbeddingBridge._to_tensors(bridge, xs, ys, ws, counts)
    feats = _features(data.x, bridge._embed(data.w_envs)[data.env])
    out = data.env == (-1 if held is None else held)
    A = _weighted_ridge(feats[~out], data.y[~out], data.c[~out], bridge.reg)
    scored = out if held is not None else ~out
    return _weighted_sse(feats[scored] @ A, data.y[scored], data.c[scored]), A


def test_per_x_bridge_is_the_shared_bridge_with_phi_the_identity():
    rng = np.random.default_rng(0)
    xs, ys, ws, counts = _categorical_envs(rng)
    bridge = NeuralPerXEmbeddingBridge(n_steps=2, n_features_w=2, reg=1e-2)
    bridge.fit(xs, ys, ws, counts)
    with torch.no_grad():
        loss, A = bridge._stage2(bridge._to_tensors(xs, ys, ws, counts))
        sse, dense_A = _dense_fit(bridge, xs, ys, ws, counts)
    torch.testing.assert_close(loss, sse / sum(c.sum() for c in counts))
    torch.testing.assert_close(A.flatten(0, 1), dense_A)
    torch.testing.assert_close(bridge.A_, A)


def test_per_x_cross_fit_with_one_fold_per_environment_is_leave_one_environment_out():
    rng = np.random.default_rng(0)
    xs, ys, ws, counts = _categorical_envs(rng)
    bridge = NeuralPerXEmbeddingBridge(n_steps=2, n_features_w=2, reg=1e-2)
    bridge.fit(xs, ys, ws, counts)
    bridge.cross_fit_folds = 4
    with torch.no_grad():
        loss = bridge._cross_fit_loss(bridge._to_tensors(xs, ys, ws, counts))
        sse = sum(_dense_fit(bridge, xs, ys, ws, counts, held=e)[0] for e in range(4))
    torch.testing.assert_close(loss, sse / sum(c.sum() for c in counts))


def test_per_x_bridge_predicts_each_category_from_its_own_A():
    rng = np.random.default_rng(0)
    xs, ys, ws, counts = _categorical_envs(rng)
    bridge = NeuralPerXEmbeddingBridge(n_steps=2, n_features_w=2).fit(xs, ys, ws, counts)
    w = rng.normal(size=(5, 2))
    with torch.no_grad():
        mu = bridge.psi_(torch.as_tensor(w)).mean(dim=0)
    pred = bridge.predict(np.eye(3)[[2, 0]], w)
    np.testing.assert_allclose(pred, (mu @ bridge.A_[[2, 0]]).numpy())


def test_neural_bridge_transfer_per_perturbation_uses_the_per_x_bridge():
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    perts = sorted(set(test.obs["perturbation"]) - {"control"})
    model = NeuralBridgeTransfer(per_perturbation=True, n_features_w=3, n_steps=2).fit(train)
    assert isinstance(model.bridge_, NeuralPerXEmbeddingBridge)
    assert model.predict(perts, test).shape == (len(perts), 5)
