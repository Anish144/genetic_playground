import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def recovery_data():
    rows, values = [], []
    for e in range(7):
        for label, offset, present in [
            ("control", 0, True),
            ("a", 10, e < 6),
            ("b", 20, 1 <= e < 6),
            ("c", 30, e < 4),
        ]:
            if not present:
                continue
            count = e + 2 if label == "control" else e + 1
            for _ in range(count):
                rows.append(
                    (
                        label,
                        f"HUVEC-{e}",
                        "HUVEC",
                        1,
                        "negative_control" if label == "control" else "treatment",
                    )
                )
                values.append([offset + e, 2 * offset - e])
    obs = pd.DataFrame(rows, columns=["perturbation", "batch", "cell_type", "plate", "well_type"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.array(values, dtype=float), obs=obs, var=pd.DataFrame(index=["f0", "f1"]))


@pytest.fixture
def sparse():
    return importlib.import_module(
        "genetic_perturbation_playground.experiments.rxrx1_sparse_recovery"
    )


def response_pairs(data):
    return set(
        data.obs.loc[data.obs.perturbation != "control", ["perturbation", "batch"]].itertuples(
            index=False, name=None
        )
    )


def test_retention_is_nested_and_row_order_invariant(sparse, recovery_data):
    pairs = sparse.reference_pairs(recovery_data)
    order = sparse.retained_order(pairs, 17)
    pd.testing.assert_frame_equal(order, sparse.retained_order(pairs.iloc[::-1], 17))
    retained = [response_pairs(sparse.retained_data(recovery_data, order, k)) for k in [2, 3, 4]]
    assert retained[0] < retained[1] < retained[2]
    assert len(retained[0]) == 6
    assert not sparse.retained_order(pairs, 18).equals(order)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_inner_folds_keep_k_minus_one_response_batches_and_all_controls(sparse, recovery_data, k):
    order = sparse.retained_order(sparse.reference_pairs(recovery_data), 0)
    controls = set(recovery_data.obs_names[recovery_data.obs.perturbation == "control"])
    held_out = []

    class Recorder:
        def predict_environment_average(self, labels, reference):
            assert set(reference.obs.perturbation) == {"control"}
            assert reference.obs.batch.nunique() == 1
            held_out.extend((p, reference.obs.batch.iloc[0]) for p in labels)
            return np.zeros((len(labels), 2))

    folds = list(sparse.inner_folds(recovery_data, order, k))
    assert len(folds) == k
    for train, evaluate in folds:
        responses = train.obs[train.obs.perturbation != "control"]
        assert responses.groupby("perturbation").batch.nunique().eq(k - 1).all()
        assert set(train.obs_names[train.obs.perturbation == "control"]) == controls
        assert np.isfinite(evaluate(Recorder())["mse"])
    assert len(held_out) == 3 * k
    assert set(held_out) == response_pairs(sparse.retained_data(recovery_data, order, k))


def test_reference_means_weight_batches_equally(sparse, recovery_data):
    full = sparse.observed_average(recovery_data, ["a", "b", "c"])
    np.testing.assert_allclose(full, [[12.5, 17.5], [23, 37], [31.5, 58.5]])
    kept = recovery_data[
        (recovery_data.obs.perturbation == "a")
        & recovery_data.obs.batch.isin(["HUVEC-0", "HUVEC-2"])
    ]
    retained = sparse.observed_average(kept, ["a"])
    np.testing.assert_allclose(retained, [[11, 19]])
    coverage = sparse.eligibility_table(sparse.reference_pairs(recovery_data), [2])
    scores = sparse.score_estimates(
        full.loc[["a"]],
        retained,
        {"kernel_virtual": full.loc[["a"]]},
        coverage[coverage.perturbation == "a"],
        seed=0,
        k=2,
    )
    assert scores.set_index("model").value.to_dict() == pytest.approx(
        {"retained_mean": 2.25, "kernel_virtual": 0}
    )


def test_virtual_average_uses_each_original_reference(sparse, recovery_data):
    class Oracle:
        perturbations_ = np.array(["a", "b", "c"])
        feature_names_in_ = np.array(["f0", "f1"])

        def predict_environment_average(self, labels, reference, **kwargs):
            actual = set(reference.obs.batch)
            for p in labels:
                expected = {"a": range(6), "b": range(1, 6), "c": range(4)}[p]
                assert actual == {f"HUVEC-{e}" for e in expected}
            e = np.mean([int(b.split("-")[1]) for b in actual])
            return np.array(
                [[o + e, 2 * o - e] for o in [{"a": 10, "b": 20, "c": 30}[p] for p in labels]]
            )

    actual = sparse.virtual_average(
        Oracle(),
        recovery_data[recovery_data.obs.perturbation == "control"],
        sparse.reference_pairs(recovery_data),
    )
    np.testing.assert_allclose(actual, [[12.5, 17.5], [23, 37], [31.5, 58.5]])


def test_eligibility_never_silently_reduces_k(sparse, recovery_data):
    pairs = sparse.reference_pairs(recovery_data)
    eligible = sparse.eligibility_table(pairs, [4, 5, 6])
    assert eligible[eligible.k == 5].set_index("perturbation").eligible.to_dict() == {
        "a": True,
        "b": True,
        "c": False,
    }
    train = sparse.retained_data(recovery_data, sparse.retained_order(pairs, 0), 6)
    assert set(train.obs.perturbation) == {"control", "a"}
    np.testing.assert_allclose(sparse.observed_average(train, ["a"]), [[12.5, 17.5]])


def test_removed_outcomes_cannot_change_inner_training_or_validation(sparse, recovery_data):
    order = sparse.retained_order(sparse.reference_pairs(recovery_data), 0)
    kept = sparse.retained_data(recovery_data, order, 2)
    poisoned = recovery_data.copy()
    poisoned.X[~poisoned.obs_names.isin(kept.obs_names)] = 1e10

    class Zero:
        def predict_environment_average(self, labels, reference):
            return np.zeros((len(labels), 2))

    for (train, evaluate), (other, other_evaluate) in zip(
        sparse.inner_folds(recovery_data, order, 2),
        sparse.inner_folds(poisoned, order, 2),
        strict=True,
    ):
        assert train.obs_names.equals(other.obs_names)
        np.testing.assert_array_equal(train.X, other.X)
        assert evaluate(Zero()) == other_evaluate(Zero())


def test_observed_average_rejects_missing_and_nonfinite_values(sparse, recovery_data):
    with pytest.raises(ValueError, match="missing"):
        sparse.observed_average(recovery_data, ["absent"])
    recovery_data.X[recovery_data.obs.perturbation == "a"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sparse.observed_average(recovery_data, ["a"])


def test_score_estimates_requires_matching_features(sparse, recovery_data):
    full = sparse.observed_average(recovery_data, ["a"])
    with pytest.raises(ValueError, match="align"):
        sparse.score_estimates(full, full.iloc[:, ::-1], {}, pd.DataFrame(), seed=0, k=2)
