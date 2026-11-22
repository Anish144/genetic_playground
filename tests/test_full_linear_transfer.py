"""Compare full transfer with independent cell-level least squares."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from genetic_perturbation_playground.model import linear_transfer


def _data(controls, counts, intercept, coefficient, *, noise=0.0):
    rng = np.random.default_rng(9)
    rows, values = [], []
    for e, control in enumerate(controls):
        # Nonzero within-batch variation also exercises control standardisation.
        for offset in (-0.2, 0.2):
            rows.append(("control", f"e{e}"))
            values.append(control + offset)
        for x, count in enumerate(counts[:, e]):
            for _ in range(count):
                rows.append((f"p{x}", f"e{e}"))
                values.append(
                    intercept[x] + coefficient @ control + noise * rng.normal(size=len(control))
                )
    obs = pd.DataFrame(rows, columns=["perturbation", "batch"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.array(values), obs=obs)


def _direct_fit(data, model):
    """Residualise individual response cells, then solve augmented least squares."""
    controls = data.obs.perturbation.eq("control").to_numpy()
    responses = data[~controls]
    labels = responses.obs.perturbation.to_numpy()
    means = {
        batch: data.X[controls & data.obs.batch.eq(batch).to_numpy()].mean(0)
        for batch in data.obs.batch.unique()
    }
    proxy = np.stack([means[batch] for batch in responses.obs.batch])
    proxy = (proxy - model.w_mean_) / model.w_std_
    target = responses.X.copy()
    control_means, target_means = [], []
    for label in model.perturbations_:
        mask = labels == label
        control_means.append(proxy[mask].mean(0))
        target_means.append(target[mask].mean(0))
        proxy[mask] -= control_means[-1]
        target[mask] -= target_means[-1]
    penalty = np.sqrt(len(responses) * model.reg) * np.eye(data.n_vars)
    design = np.vstack([proxy, penalty])
    target = np.vstack([target, np.zeros((data.n_vars, data.n_vars))])
    coefficient = np.linalg.lstsq(design, target, rcond=None)[0].T
    intercept = np.array(target_means) - np.array(control_means) @ coefficient.T
    return coefficient, intercept


@pytest.mark.parametrize("standardise_w", [False, True])
def test_full_transfer_recovers_cross_coordinate_effects(standardise_w):
    coefficient = np.array([[0.0, 3.0], [-2.0, 0.0]])
    intercept = np.array([[5.0, 7.0], [-4.0, 2.0]])
    train = _data(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]]),
        np.array([[1, 3, 2, 4], [2, 0, 1, 3]]),
        intercept,
        coefficient,
    )
    model = linear_transfer.FullLinearTransfer(standardise_w=standardise_w, reg=0).fit(train)
    test = _data(np.array([[3.0, 5.0]]), np.ones((2, 1), dtype=int), intercept, coefficient)
    test.X[test.obs.perturbation.ne("control")] = np.nan
    np.testing.assert_allclose(model.slope_, coefficient, atol=1e-12)
    np.testing.assert_allclose(model.gamma_, intercept, atol=1e-12)
    np.testing.assert_allclose(
        model.predict(["p1", "p0", "p1"], test),
        [[11.0, -4.0], [20.0, 1.0], [11.0, -4.0]],
        atol=1e-12,
    )


@pytest.mark.parametrize("standardise_w", [False, True])
@pytest.mark.parametrize("reg", [0.0, 0.001, 0.7])
@pytest.mark.parametrize("n_features", [2, 9])
def test_full_transfer_matches_cell_level_ridge(standardise_w, reg, n_features):
    rng = np.random.default_rng(14)
    train = _data(
        rng.normal(size=(5, n_features)),
        np.array([[3, 0, 1, 2, 4], [0, 2, 4, 1, 1], [2, 1, 0, 0, 3]]),
        rng.normal(size=(3, n_features)),
        rng.normal(size=(n_features, n_features)),
        noise=0.3,
    )
    model = linear_transfer.FullLinearTransfer(standardise_w=standardise_w, reg=reg).fit(train)
    coefficient, intercept = _direct_fit(train, model)
    np.testing.assert_allclose(model.coef_, coefficient, atol=1e-10)
    np.testing.assert_allclose(model.intercept_, intercept, atol=1e-10)
    np.testing.assert_allclose(model.slope_, coefficient / model.w_std_, atol=1e-10)
    np.testing.assert_allclose(
        model.gamma_,
        intercept - model.w_mean_ @ (coefficient / model.w_std_).T if standardise_w else intercept,
        atol=1e-10,
    )


@pytest.mark.parametrize("reg", [0.0, 0.3])
def test_full_transfer_without_within_category_variation_returns_pooled_means(reg):
    train = _data(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[3, 0], [0, 2]]),
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        np.eye(2),
    )
    model = linear_transfer.FullLinearTransfer(reg=reg).fit(train)
    np.testing.assert_array_equal(model.coef_, np.zeros((2, 2)))
    np.testing.assert_allclose(model.predict(["p0", "p1"], train), [[11.0, 22.0], [33.0, 44.0]])


def test_full_transfer_handles_collinear_and_constant_controls():
    train = _data(
        np.array([[0.0, 0.0, 8.0], [1.0, 2.0, 8.0], [2.0, 4.0, 8.0]]),
        np.array([[1, 2, 3]]),
        np.zeros((1, 3)),
        np.eye(3),
    )
    model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=0).fit(train)
    np.testing.assert_allclose(
        model.slope_, [[0.2, 0.4, 0.0], [0.4, 0.8, 0.0], [0.0, 0.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(model.gamma_, [[0.0, 0.0, 8.0]], atol=1e-12)


def test_full_transfer_cannot_learn_between_disconnected_categories():
    train = _data(
        np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]),
        np.array([[3, 1, 0, 0], [0, 0, 2, 3]]),
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        np.eye(2),
    )
    model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=0).fit(train)
    np.testing.assert_allclose(model.coef_, np.zeros((2, 2)), atol=1e-12)
    np.testing.assert_allclose(model.gamma_, [[11.0, 22.0], [33.0, 44.0]], atol=1e-12)


@pytest.mark.parametrize("reg", [0.0, 0.3])
def test_full_transfer_ignores_large_global_offsets(reg):
    controls = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]])
    counts = np.array([[1, 3, 2, 4], [2, 0, 1, 3]])
    intercept = np.array([[5.0, 7.0], [-4.0, 2.0]])
    coefficient = np.array([[0.0, 3.0], [-2.0, 0.0]])
    train = _data(controls, counts, intercept, coefficient)
    shifted = train.copy()
    shifted.X += 1e8
    model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=reg).fit(train)
    shifted_model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=reg).fit(shifted)
    np.testing.assert_allclose(shifted_model.coef_, model.coef_, atol=1e-8)


@pytest.mark.parametrize("reg", [0.0, 0.001])
def test_full_transfer_ignores_control_only_batches_without_standardisation(reg):
    train = _data(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1, 1]]),
        np.zeros((1, 2)),
        np.array([[2.0, 0.0], [2.0, 0.0]]),
    )
    extra = _data(
        np.array([[0.0, 1e16]]), np.zeros((1, 1), dtype=int), np.zeros((1, 2)), np.zeros((2, 2))
    )
    extra.obs["batch"] = "unused"
    model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=reg).fit(train)
    augmented = linear_transfer.FullLinearTransfer(standardise_w=False, reg=reg).fit(
        ad.concat([train, extra], index_unique="-")
    )
    np.testing.assert_allclose(augmented.coef_, model.coef_, atol=1e-12)
    np.testing.assert_allclose(augmented.predict(["p0"], train), model.predict(["p0"], train))


@pytest.mark.parametrize("reg", [0.0, 0.001])
def test_full_transfer_ignores_offsets_between_disconnected_components(reg):
    train = _data(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1e16], [1.0, 1e16]]),
        np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
        np.array([[5.0, 7.0], [8.0, 3.0]]),
        np.array([[2.0, 0.0], [2.0, 0.0]]),
    )
    model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=reg).fit(train)
    np.testing.assert_allclose(model.coef_, [[2 / (1 + 4 * reg), 0.0]] * 2, atol=1e-12)


def test_full_transfer_preserves_sparse_inputs_and_count_replication(virtual_train):
    original = virtual_train.X.copy()
    model = linear_transfer.FullLinearTransfer(reg=0.2).fit(virtual_train)
    replicated = ad.concat([virtual_train] * 3, index_unique="-")
    replicated.X = sparse.csr_matrix(replicated.X)
    repeated = linear_transfer.FullLinearTransfer(reg=0.2).fit(replicated)
    np.testing.assert_allclose(
        repeated.predict(["b", "a"], replicated), model.predict(["b", "a"], virtual_train)
    )
    np.testing.assert_array_equal(virtual_train.X, original)


@pytest.mark.parametrize("reg", [-1.0, np.nan, np.inf])
def test_full_transfer_rejects_invalid_penalties(virtual_train, reg):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        linear_transfer.FullLinearTransfer(reg=reg).fit(virtual_train)


def test_full_transfer_requires_responses(virtual_train):
    controls = virtual_train[virtual_train.obs.perturbation.eq("control")]
    with pytest.raises(ValueError, match="positive total response weight"):
        linear_transfer.FullLinearTransfer().fit(controls)


@pytest.mark.parametrize("reg", [0.0, 1e-33])
def test_full_transfer_retains_small_control_units_with_one_output(reg):
    obs = pd.DataFrame(
        {"perturbation": ["control", "p0", "control", "p0"], "batch": ["e0", "e0", "e1", "e1"]},
        index=["0", "1", "2", "3"],
    )
    train = ad.AnnData(np.array([[0.0], [0.0], [2e-16], [6.0]]), obs=obs)
    model = linear_transfer.FullLinearTransfer(standardise_w=False, reg=reg).fit(train)
    expected = 3e16 / (1 + reg / 1e-32)
    np.testing.assert_allclose(model.slope_, [[expected]])
    np.testing.assert_allclose(model.gamma_, [[3 - expected * 1e-16]], atol=1e-14)
