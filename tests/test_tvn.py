import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from genetic_perturbation_playground.model.batch_transfer import BatchTransferModel
from genetic_perturbation_playground.model.tvn import TVNTransferModel


def _affine_batches(rng, n_ctrl, n_features=4):
    """Controls z ~ N(0, I), treatments z + delta_x, observed as z A_e + c_e, A_e = V diag(a_e) V^T:
    shared axes V, per-batch scales a_e (axes 2x apart, so the pooled basis identifies V)."""
    deltas = {"a": np.full(n_features, 1.0), "b": np.linspace(-1, 1, n_features)}
    V, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    rows, X = [], []
    for e in range(3):
        A = (V * 2.0 ** np.arange(n_features) * rng.uniform(0.8, 1.25, n_features)) @ V.T
        c = rng.normal(size=n_features) * 5
        for label, shift in [("control", 0.0), *deltas.items()]:
            X.append((rng.normal(size=(n_ctrl, n_features)) + shift) @ A + c)
            rows += [(label, f"b{e}")] * n_ctrl
    obs = pd.DataFrame(rows, columns=["perturbation", "batch"])
    truth = np.stack([deltas[x] @ A + c for x in ["a", "b"]])  # means in the last batch, b2
    return ad.AnnData(np.vstack(X), obs=obs), truth


def _recursion_tvn(X, ctrl, batch):
    """tvn_on_controls from recursionpharma/EFAAR_benchmarking (efaar_benchmarking/efaar.py)."""
    X = StandardScaler().fit(X[ctrl]).transform(X)
    X = PCA().fit(X[ctrl]).transform(X)
    for b in np.unique(batch):
        X[batch == b] = StandardScaler().fit(X[(batch == b) & ctrl]).transform(X[batch == b])
    target_cov = np.cov(X[ctrl], rowvar=False, ddof=1) + 0.5 * np.eye(X.shape[1])
    for b in np.unique(batch):
        source_cov = np.cov(X[(batch == b) & ctrl], rowvar=False, ddof=1) + 0.5 * np.eye(X.shape[1])
        X[batch == b] = X[batch == b] @ linalg.fractional_matrix_power(source_cov, -0.5)
        X[batch == b] = X[batch == b] @ linalg.fractional_matrix_power(target_cov, 0.5)
    return X


def test_tvn_profiles_match_recursion_tvn():
    data, _ = _affine_batches(np.random.default_rng(0), n_ctrl=50)
    ctrl = (data.obs["perturbation"] == "control").to_numpy()
    normalised = _recursion_tvn(data.X, ctrl, data.obs["batch"].to_numpy())
    perts = data.obs["perturbation"].to_numpy()
    expected = np.stack([normalised[perts == p].mean(0) for p in ["a", "b"]])
    np.testing.assert_allclose(TVNTransferModel().fit(data).profiles_, expected, atol=1e-8)


def test_tvn_recovers_affine_batch_effects():
    data, truth = _affine_batches(np.random.default_rng(0), n_ctrl=20_000)
    train, test = data[data.obs["batch"] != "b2"], data[data.obs["batch"] == "b2"]
    pred = TVNTransferModel(ridge=0.0).fit(train).predict(["a", "b"], test)
    np.testing.assert_allclose(pred, truth, rtol=0.02, atol=0.1)


def test_tvn_matches_shift_on_shift_only_batches_with_all_controls():
    """Pure shift batch effects with RxRx1's negative and positive controls (~4,000 per batch, 384
    features). With the negatives alone (~128) TVN's per-batch covariance is rank deficient and its
    error is ~50x shift's, so that regime is not asserted."""
    rng = np.random.default_rng(0)
    n_features, n_ctrl, n_treated = 384, 4000, 32
    effect = rng.normal(size=n_features)
    rows, X = [], []
    for e in range(6):
        offset = rng.normal(size=n_features) * 3
        X.append(rng.normal(size=(n_ctrl, n_features)) + offset)
        X.append(rng.normal(size=(n_treated, n_features)) + offset + effect)
        rows += [("control", f"b{e}")] * n_ctrl + [("a", f"b{e}")] * n_treated
    data = ad.AnnData(np.vstack(X), obs=pd.DataFrame(rows, columns=["perturbation", "batch"]))
    train, test = data[data.obs["batch"] != "b5"], data[data.obs["batch"] == "b5"]
    truth = test.X[(test.obs["perturbation"] == "a").to_numpy()].mean(0)

    def mse(model):
        return np.mean((model.fit(train).predict(["a"], test)[0] - truth) ** 2)

    assert mse(TVNTransferModel()) < 1.5 * mse(BatchTransferModel(slope=1.0))


def test_tvn_rejects_several_test_batches():
    data, _ = _affine_batches(np.random.default_rng(0), n_ctrl=50)
    with pytest.raises(ValueError, match="one batch"):
        TVNTransferModel().fit(data).predict(["a"], data)
