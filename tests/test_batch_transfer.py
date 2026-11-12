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
        weighted.predict(x_new, ws[0]),
        replicated.predict(x_new, ws[0]),
        rtol=1e-4,  # float32 nets
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


@pytest.mark.parametrize("per_perturbation", [False, True])
@pytest.mark.parametrize("cross_fit_folds", [0, 2])
def test_neural_checkpoint_equals_fit_with_that_many_steps(per_perturbation, cross_fit_folds):
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    perts = sorted(set(test.obs["perturbation"]) - {"control"})
    kwargs = {
        "n_features": 4,
        "n_features_w": 3,
        "n_steps": 6,
        "per_perturbation": per_perturbation,
        "cross_fit_folds": cross_fit_folds,
    }
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
    first, second, out = mlp.net
    x = torch.randn(6, 3)

    def swiglu(layer, h):
        return torch.nn.functional.silu(layer.gate(h)) * layer.value(h)

    h1 = swiglu(first, x)  # 3 -> 4 units: no residual
    h2 = h1 + swiglu(second.layer, h1)  # 4 -> 4 units: residual
    torch.testing.assert_close(mlp(x.double()), out(h2).double())  # float32 inside


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


@pytest.mark.parametrize("gene_block_size", [1, 3, 256])
def test_per_x_bridge_is_the_shared_bridge_with_phi_the_identity(gene_block_size):
    rng = np.random.default_rng(0)
    # G = q = D = 4 exposes ambiguity between matrix and batched-vector ridge RHS shapes.
    xs, ys, ws, counts = _categorical_envs(rng, n_cats=4, d_y=7)
    bridge = NeuralPerXEmbeddingBridge(
        n_steps=2, n_features_w=4, reg=1e-2, gene_block_size=gene_block_size
    )
    bridge.fit(xs, ys, ws, counts)
    with torch.no_grad():
        loss = bridge._training_loss(bridge._to_tensors(xs, ys, ws, counts))
        sse, dense_A = _dense_fit(bridge, xs, ys, ws, counts)
    torch.testing.assert_close(loss, sse / sum(c.sum() for c in counts))
    torch.testing.assert_close(bridge.A_.flatten(0, 1), dense_A)


@pytest.mark.parametrize("cross_fit_folds", [0, 2, 4])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n_cats, n_features_w", [(3, 2), (3, 6), (4, 4)])
def test_per_x_target_gram_loss_and_gradients_match_dense_ridge(
    monkeypatch, cross_fit_folds, weighted, n_cats, n_features_w
):
    """Compare against explicit pair predictions, including a rank-deficient feature Gram.

    G=n_cats categories, D=4 environments, J=7 genes, q=n_features_w features. Category 0
    occurs only in environment 0, so some folds have no training data for that category.
    Environment 2 has proxies but no pairs. Category 1 has two DIFFERENT targets in
    environment 0, testing cyy's within-group error. G=q=D=4 checks matrix-RHS disambiguation.
    """
    rng = np.random.default_rng(12)
    xs, ys, ws, counts = _categorical_envs(rng, n_cats=n_cats, d_y=7)
    for e in [1, 2]:
        xs[e], ys[e], counts[e] = xs[e][1:], ys[e][1:], counts[e][1:]
    xs[2], ys[2], counts[2] = xs[2][:0], ys[2][:0], counts[2][:0]
    xs[0] = np.concatenate([xs[0], xs[0][[1]]])
    ys[0] = np.concatenate([ys[0], rng.normal(size=(1, 7))])
    counts[0] = np.append(counts[0], 5)
    if not weighted:
        counts = None
    torch.manual_seed(12)
    bridge = NeuralPerXEmbeddingBridge(
        n_features_w=n_features_w, width_w=4, depth=1, cross_fit_folds=cross_fit_folds, reg=1e-5
    )
    data = bridge._to_tensors(xs, ys, ws, counts)
    bridge.psi_ = _mlp(2, 4, 1, n_features_w)
    mu = bridge._embed(data.w_envs)  # (D, q), connected to the float32 network
    monkeypatch.setattr(bridge, "_embed", lambda _: mu)
    fold = torch.arange(4) % max(cross_fit_folds, 1)  # (D,)
    monkeypatch.setattr(bridge, "_env_folds", lambda _: fold)

    assert data.cy.shape == (n_cats, 4, 7)  # Z (G, D, J), needed only for prediction
    assert all(
        t.device == mu.device for t in (data.c, data.cy, data.cyy, data.target_gram, *data.w_envs)
    )
    assert data.target_gram.shape == (n_cats, 4, 4)  # Q (G, D, D), sufficient for training
    # Removing the gene targets must not affect training, including checkpoint recomputation.
    compressed = bridge._training_loss(data._replace(cy=torch.empty(0)))

    pairs = NeuralMarginalEmbeddingBridge._to_tensors(bridge, xs, ys, ws, counts)
    feats = _features(pairs.x, mu[pairs.env])  # (N, G*q), one feature row per observed pair
    reference = 0.0
    for k in range(max(cross_fit_folds, 1)):
        scored = fold[pairs.env] == k
        train = ~scored if cross_fit_folds else scored
        A = _weighted_ridge(feats[train], pairs.y[train], pairs.c[train], bridge.reg)  # (G*q, J)
        residual = feats[scored] @ A - pairs.y[scored]  # (N_scored, J), possibly zero rows
        reference = reference + (pairs.c[scored] * residual.square().mean(dim=1)).sum()
    reference = reference / pairs.c.sum()
    torch.testing.assert_close(compressed, reference, rtol=1e-9, atol=1e-10)

    variables = (mu, *bridge.psi_.parameters())
    actual_grad = torch.autograd.grad(compressed, variables, retain_graph=True)
    expected_grad = torch.autograd.grad(reference, variables)
    torch.testing.assert_close(actual_grad[0], expected_grad[0], rtol=1e-8, atol=1e-9)
    for actual, expected in zip(actual_grad[1:], expected_grad[1:], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)  # float32 NN backward


@pytest.mark.parametrize("gene_block_size", [0, -1])
def test_per_x_reconstruction_needs_positive_gene_block_size(gene_block_size):
    xs, ys, ws, counts = _categorical_envs(np.random.default_rng(0))
    with pytest.raises(ValueError, match="gene_block_size must be positive"):
        NeuralPerXEmbeddingBridge(gene_block_size=gene_block_size).fit(xs, ys, ws, counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_per_x_cuda_training_and_block_reconstruction_stay_on_device():
    xs, ys, ws, counts = _categorical_envs(np.random.default_rng(0), d_y=7)
    bridge = NeuralPerXEmbeddingBridge(
        n_steps=2, n_features_w=6, cross_fit_folds=2, gene_block_size=3, device="cuda"
    ).fit(xs, ys, ws, counts)
    data = bridge._to_tensors(xs, ys, ws, counts)
    assert all(
        t.device.type == "cuda" for t in (data.c, data.cy, data.cyy, data.target_gram, *data.w_envs)
    )
    assert bridge.A_.device.type == "cuda"
    with torch.no_grad():
        sse, dense_A = _dense_fit(bridge, xs, ys, ws, counts)
        mu = bridge.psi_(torch.as_tensor(ws[0], device="cuda")).mean(dim=0)
        expected = mu @ dense_A.reshape(3, 6, 7)  # A: (G=3, q=6, J=7)
    torch.testing.assert_close(bridge.A_.flatten(0, 1), dense_A, rtol=1e-8, atol=1e-9)
    assert torch.isfinite(sse)
    np.testing.assert_allclose(bridge.predict(np.eye(3), ws[0]), expected.cpu().numpy())


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
