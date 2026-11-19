import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import wandb
from neural_reference import features as _features
from neural_reference import pair_data
from neural_reference import weighted_ridge as _weighted_ridge
from neural_reference import weighted_sse as _weighted_sse

from genetic_perturbation_playground.data.transfer import (
    batch_controls,
    bridge_data,
)
from genetic_perturbation_playground.model.batch_transfer import (
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
)
from genetic_perturbation_playground.model.linear_transfer import (
    LinearTransfer,
)
from genetic_perturbation_playground.model.proxy_unconfounded import MarginalEmbeddingBridge
from genetic_perturbation_playground.model.proxy_unconfounded_neural import (
    NeuralHybridEmbeddingBridge,
    NeuralPerXEmbeddingBridge,
    NeuralSharedEmbeddingBridge,
)
from genetic_perturbation_playground.utils.kernels import RBF
from genetic_perturbation_playground.utils.neural import control_mlp
from genetic_perturbation_playground.utils.standardisation import (
    Standardised,
)


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


@pytest.mark.parametrize("prepare", [bridge_data, batch_controls])
def test_bridge_data_requires_controls_in_every_perturbed_batch(prepare):
    data = ad.AnnData(
        np.ones((2, 3)),
        obs=pd.DataFrame({"perturbation": ["control", "a"], "batch": ["b0", "b1"]}),
    )
    with pytest.raises(ValueError, match="Every batch with perturbed cells needs control cells"):
        prepare(data)


@pytest.mark.parametrize("standardise_w", [True, False])
@pytest.mark.parametrize("slope, expected", [(0, [45, 397]), (1, [75, 487]), (None, [105, 667])])
def test_linear_input_scaling_preserves_raw_targets_and_fixed_slopes(
    standardise_w, slope, expected
):
    # Y = [5, 7] + [2, 3] * W. Each batch has two controls and one perturbed cell.
    X = np.array([[8, 90], [12, 110], [25, 307], [28, 150], [32, 170], [65, 487]], dtype=float)
    obs = pd.DataFrame(
        {"perturbation": ["control", "control", "a"] * 2, "batch": ["b0"] * 3 + ["b1"] * 3}
    )
    train = ad.AnnData(X, obs=obs)
    test = ad.AnnData(
        np.array([[48, 210], [52, 230], [-999, -999]], dtype=float), obs=obs.iloc[:3].copy()
    )
    model = LinearTransfer(slope, standardise_w=standardise_w, reg=0.0).fit(train)
    std = X[[0, 1, 3, 4]].std(0) if standardise_w else 1
    raw_slope = np.array([2, 3]) if slope is None else slope
    np.testing.assert_allclose(model.coef_, raw_slope * std)
    np.testing.assert_allclose(model.predict(["a"], test), [expected])


@pytest.mark.parametrize("standardise_w", [True, False])
@pytest.mark.parametrize("reg", [0.0, 1e-3, 0.7, 10.0])
def test_linear_solver_matches_full_cell_level_ridge(standardise_w, reg):
    data = _cells(np.random.default_rng(21), n_batches=6)
    model = LinearTransfer(standardise_w=standardise_w, reg=reg).fit(data)
    controls = data.obs["perturbation"].eq("control").to_numpy()
    responses = data[~controls]
    labels = responses.obs["perturbation"].to_numpy()
    categories = np.eye(len(model.perturbations_))[np.searchsorted(model.perturbations_, labels)]
    means = {
        batch: data.X[controls & data.obs["batch"].eq(batch).to_numpy()].mean(axis=0)
        for batch in data.obs["batch"].unique()
    }
    raw_proxy = np.stack([means[batch] for batch in responses.obs["batch"]])
    proxy = (raw_proxy - model.w_mean_) / model.w_std_
    for gene in range(data.n_vars):
        design = np.column_stack((categories, proxy[:, gene]))
        # One extra row penalises only the slope. Category intercepts stay unpenalised.
        penalty_row = np.zeros((1, design.shape[1]))
        penalty_row[0, -1] = np.sqrt(len(responses) * reg)
        augmented = np.vstack((design, penalty_row))
        target = np.append(responses.X[:, gene], 0.0)
        coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(model.intercept_[:, gene], coefficients[:-1], atol=1e-10)
        np.testing.assert_allclose(model.coef_[gene], coefficients[-1], atol=1e-10)
        original_units = (
            categories @ model.gamma_[:, gene] + model.slope_[gene] * raw_proxy[:, gene]
        )
        np.testing.assert_allclose(original_units, design @ coefficients, atol=1e-10)


@pytest.mark.parametrize("slope", [0.0, 1.0])
@pytest.mark.parametrize("standardise_w", [False, True])
def test_fixed_linear_slopes_ignore_ridge_penalty(slope, standardise_w):
    data = _cells(np.random.default_rng(11))
    ordinary = LinearTransfer(slope, standardise_w, reg=0.0).fit(data)
    penalised = LinearTransfer(slope, standardise_w, reg=100.0).fit(data)
    np.testing.assert_array_equal(ordinary.coef_, penalised.coef_)
    np.testing.assert_array_equal(ordinary.intercept_, penalised.intercept_)


@pytest.mark.parametrize(
    "make_model", [LinearTransfer, KernelBridgeTransfer, lambda: NeuralBridgeTransfer(n_steps=0)]
)
@pytest.mark.parametrize("label", ["b", "z"])
def test_transfer_models_reject_unseen_perturbations(make_model, label):
    train = _cells(np.random.default_rng(0), perts=("a", "c"))
    model = make_model().fit(train)
    test = train.copy()
    test.obs.loc[test.obs["perturbation"] == "a", "perturbation"] = label
    with pytest.raises(ValueError, match="Unknown labels"):
        model.predict([label], test)


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
    xs = [np.eye(3)[rng.integers(0, 3, size=6)] for _ in range(3)]
    ys = [rng.normal(size=(6, 4)) for _ in range(3)]  # multi-output y
    ws = [rng.normal(size=(10, 3)) for _ in range(3)]
    counts = [rng.integers(1, 4, size=6) for _ in range(3)]

    def fit(xs, ys, counts=None):
        bridge = NeuralPerXEmbeddingBridge(
            n_steps=5,
            weight_decay=1e-2,
            n_features_w=3,
            cross_fit_folds=cross_fit_folds,
        )
        return bridge.fit(xs, ys, ws, counts)

    weighted = fit(xs, ys, counts)
    replicated = fit(
        [np.repeat(x, c, 0) for x, c in zip(xs, counts, strict=True)],
        [np.repeat(y, c, 0) for y, c in zip(ys, counts, strict=True)],
    )
    x_new = np.eye(3)[[0, 1, 2, 0, 1]]
    np.testing.assert_allclose(
        weighted.predict(x_new, ws[0]),
        replicated.predict(x_new, ws[0]),
        rtol=1e-4,  # float32 nets
    )


def test_score_truth_uses_only_unseen_controls():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from train_replogle_transfer import MODELS, score

    rng = np.random.default_rng(0)
    data = _cells(rng, n_batches=1, perts=("a",))
    is_ctrl = (data.obs["perturbation"] == "control").to_numpy()
    seen = np.arange(data.n_obs) % 2 == 0
    models = {n: LinearTransfer(s).fit(data) for n, s in MODELS.items()}
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


@pytest.mark.parametrize(
    "bridge_settings",
    [
        {},
        {"variant": "shared_2nn"},
        {"variant": "hybrid"},
    ],
)
@pytest.mark.parametrize("cross_fit_folds", [0, 2])
def test_neural_checkpoint_equals_fit_with_that_many_steps(bridge_settings, cross_fit_folds):
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    perts = sorted(set(test.obs["perturbation"]) - {"control"})
    kwargs = {
        "n_features_w": 3,
        "n_steps": 6,
        "cross_fit_folds": cross_fit_folds,
        **bridge_settings,
    }
    at_step = {}

    model = NeuralBridgeTransfer(**kwargs)
    model.fit(train, [0, 3], lambda step: at_step.update({step: model.predict(perts, test)}))

    assert list(at_step) == [0, 3]
    for step in at_step:
        short = NeuralBridgeTransfer(**{**kwargs, "n_steps": step}).fit(train)
        np.testing.assert_allclose(at_step[step], short.predict(perts, test))


def test_neural_bridge_logs_loss_every_step_only_with_a_log_key(monkeypatch):
    rng = np.random.default_rng(0)
    logged, axes = [], []
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", logged.append)
    monkeypatch.setattr(wandb, "define_metric", lambda key, step_metric: axes.append(step_metric))
    xs = [np.eye(3)[rng.integers(0, 3, size=5)] for _ in range(2)]
    ys, ws = ([rng.normal(size=(5, 2)) for _ in range(2)] for _ in range(2))
    NeuralPerXEmbeddingBridge(n_steps=3).fit(xs, ys, ws)
    assert logged == axes == []
    recorded = {}
    bridge = NeuralPerXEmbeddingBridge(n_steps=3, cross_fit_folds=2, log_key="fold 2/neural bridge")
    bridge.fit(xs, ys, ws, on_training_loss=recorded.__setitem__)
    assert axes == ["fold 2/neural bridge/step"]
    assert [m["fold 2/neural bridge/step"] for m in logged] == [0, 1, 2]
    assert all(np.isfinite(m["fold 2/neural bridge/loss"]) for m in logged)
    assert list(recorded) == [0, 1, 2, 3]
    assert list(recorded.values())[:3] == [m["fold 2/neural bridge/loss"] for m in logged]
    with torch.no_grad():
        final_loss = bridge._training_loss(bridge._to_tensors(xs, ys, ws, None))
    assert recorded[3] == pytest.approx(final_loss.item())


def test_cross_fit_needs_two_environments():
    rng = np.random.default_rng(0)
    bridge = NeuralPerXEmbeddingBridge(n_steps=1, cross_fit_folds=2)
    with pytest.raises(ValueError, match="two environments"):
        bridge.fit(
            [np.eye(2)[[0, 1, 0, 1, 0]]], [rng.normal(size=(5, 1))], [rng.normal(size=(4, 2))]
        )


def test_neural_bridge_sizes_control_width_and_embedding_separately():
    rng = np.random.default_rng(0)
    xs = [np.eye(3)[rng.integers(0, 3, size=5)] for _ in range(2)]
    ys, ws = ([rng.normal(size=(5, 2)) for _ in range(2)] for _ in range(2))
    bridge = NeuralPerXEmbeddingBridge(n_steps=1, width_w=5, n_features_w=6).fit(xs, ys, ws)

    def last_two_sizes(mlp):
        return [m.out_features for m in mlp.modules() if isinstance(m, torch.nn.Linear)][-2:]

    assert last_two_sizes(bridge.psi_) == [5, 6]


def test_mlp_hidden_layers_are_swiglu_with_residual_connections_where_sizes_match():
    torch.manual_seed(0)
    mlp = control_mlp(d_in=3, width=4, depth=2, d_out=5)
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
    """Fit dense one-hot × control features; score the held-out environment or all rows."""
    data = pair_data(xs, ys, ws, counts, device=bridge.device)
    feats = _features(data.x, bridge._embed(data.w_envs)[data.env])
    out = data.env == (-1 if held is None else held)
    A = _weighted_ridge(feats[~out], data.y[~out], data.c[~out], bridge.reg)
    scored = out if held is not None else ~out
    return _weighted_sse(feats[scored] @ A, data.y[scored], data.c[scored]), A


@pytest.mark.parametrize("gene_block_size", [1, 3, 256])
def test_per_x_coefficients_and_loss_match_dense_ridge(gene_block_size):
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
    bridge.psi_ = control_mlp(2, 4, 1, n_features_w)
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

    pairs = pair_data(xs, ys, ws, counts, device=bridge.device)
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


def test_neural_bridge_transfer_uses_the_selected_per_x_bridge():
    rng = np.random.default_rng(0)
    data = _cells(rng)
    train, test = data[data.obs["batch"] != "b3"], data[data.obs["batch"] == "b3"]
    perts = sorted(set(test.obs["perturbation"]) - {"control"})
    model = NeuralBridgeTransfer(variant="per_x", n_features_w=3, n_steps=2).fit(train)
    assert isinstance(model.bridge_, NeuralPerXEmbeddingBridge)
    assert model.predict(perts, test).shape == (len(perts), 5)


@pytest.mark.parametrize(
    "variant, bridge_type",
    [
        ("shared_2nn", NeuralSharedEmbeddingBridge),
        ("hybrid", NeuralHybridEmbeddingBridge),
    ],
)
def test_shared_variants_standardise_controls_and_mix_batches_without_test_response_leakage(
    variant, bridge_type
):
    data = _cells(np.random.default_rng(0), n_batches=6)
    test_mask = data.obs["batch"].isin(["b4", "b5"])
    train, test = data[~test_mask], data[test_mask].copy()
    model = Standardised(NeuralBridgeTransfer(variant=variant, n_steps=2, n_features_w=3))
    model.fit(train)
    assert isinstance(model.model.bridge_, bridge_type)
    labels = np.intersect1d(test.obs["perturbation"].unique(), model.perturbations_)
    prediction = model.predict(labels, test)
    expected = np.zeros_like(prediction)
    for row, label in enumerate(labels):
        responses = test.obs["perturbation"].eq(label)
        for batch in test.obs["batch"].unique():
            subset = test[test.obs["batch"].eq(batch)]
            controls = subset.X[subset.obs["perturbation"].eq("control")]
            controls = (controls - model.w_mean_) / model.w_std_
            x = model.model._onehot(np.searchsorted(model.perturbations_, [label]))
            share = (responses & test.obs["batch"].eq(batch)).sum() / responses.sum()
            expected[row] += share * model.model.bridge_.predict(x, controls)[0]
    np.testing.assert_allclose(prediction, expected + model.y_mean_)
    test.X[test.obs["perturbation"].ne("control")] += 10000
    np.testing.assert_array_equal(model.predict(labels, test), prediction)


def test_neural_variant_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown neural variant"):
        NeuralBridgeTransfer(variant="typo")


def test_marginal_neural_variant_has_been_removed():
    with pytest.raises(ValueError, match="Unknown neural variant"):
        NeuralBridgeTransfer(variant="marginal")


def test_neural_transfer_defaults_to_per_x():
    train = _cells(np.random.default_rng(0))
    model = NeuralBridgeTransfer(n_steps=0).fit(train)
    assert type(model.bridge_) is NeuralPerXEmbeddingBridge
