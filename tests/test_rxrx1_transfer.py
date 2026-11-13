import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from genetic_perturbation_playground.model.batch_transfer import Standardised

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_batch_transfer_rxrx1 as mod  # noqa: E402
import tune_bridges_rxrx1 as tune  # noqa: E402


def cfg(**overrides):
    """The script's config with `overrides`, in struct mode as Hydra hands it over."""
    config = OmegaConf.load(ROOT / "configs" / "train_batch_transfer_rxrx1.yaml")
    OmegaConf.set_struct(config, True)
    return OmegaConf.merge(config, {"name": "test", "wandb_mode": "disabled"}, overrides)


@pytest.fixture
def tiny_rxrx1():
    """3 HUVEC and 1 HEPG2 experiments: 30 negative-control crops, positive control p1 and
    siRNAs s1, s2 (10 crops each)."""
    rng = np.random.default_rng(0)
    sirna_mean = {"control": 0.0, "p1": 2.0, "s1": 1.0, "s2": -1.0}
    well_type = {"control": "negative_control", "p1": "positive_control"}
    rows, X = [], []
    for batch in ["HUVEC-1", "HUVEC-2", "HUVEC-3", "HEPG2-1"]:
        offset = rng.normal(size=3) * 3
        for label, n in [("control", 30), ("p1", 10), ("s1", 10), ("s2", 10)]:
            X.append(sirna_mean[label] + offset + rng.normal(size=(n, 3)))
            rows += [(label, batch, batch.split("-")[0], well_type.get(label, "treatment"))] * n
    obs = pd.DataFrame(rows, columns=["perturbation", "batch", "cell_type", "well_type"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.vstack(X), obs=obs)


def test_select_cell_type(tiny_rxrx1):
    assert set(mod.select_cell_type(tiny_rxrx1, "HUVEC").obs["cell_type"]) == {"HUVEC"}
    with pytest.raises(ValueError, match="HUVCE"):
        mod.select_cell_type(tiny_rxrx1, "HUVCE")


def test_all_cell_types_make_each_cell_type_and_sirna_one_perturbation(tiny_rxrx1):
    labels = set(mod.select_cell_type(tiny_rxrx1, None).obs["perturbation"])
    assert labels == {"control"} | {
        f"{c}|{s}" for c in ["HUVEC", "HEPG2"] for s in ["p1", "s1", "s2"]
    }


def test_folds_over_all_cell_types_keep_every_cell_type_in_training():
    batches = [f"{c}-{i}" for c, n in [("HUVEC", 6), ("U2OS", 2)] for i in range(n)]
    folds = mod.experiment_folds(np.array(batches), 2, np.random.default_rng(0), cell_type=None)
    assert [sorted(b.split("-")[0] for b in f) for f in folds] == [["HUVEC"] * 3 + ["U2OS"]] * 2


def test_mse_of_perfect_prediction_is_zero(tiny_rxrx1):
    test = tiny_rxrx1[tiny_rxrx1.obs["batch"] == "HUVEC-1"]
    obs_means = {
        p: test.X[(test.obs["perturbation"] == p).to_numpy()].mean(0) for p in ["s1", "s2"]
    }

    class Oracle:
        perturbations_ = np.array(["s1", "s2"])

        def predict(self, perts, test):
            return np.stack([obs_means[p] for p in perts])

    scores = mod.score_batch({"oracle": Oracle()}, test)
    np.testing.assert_allclose(scores["value"], 0.0, atol=1e-12)
    assert set(scores["metric"]) == {"mse"}


def test_positive_controls_are_never_scored(tiny_rxrx1):
    test = tiny_rxrx1[tiny_rxrx1.obs["batch"] == "HUVEC-1"]

    class Zero:
        perturbations_ = np.array(["p1", "s1", "s2"])

        def predict(self, perts, test):
            return np.zeros((len(perts), test.n_vars))

    scores = mod.score_batch({"zero": Zero()}, test)
    assert set(scores["perturbation"]) == {"s1", "s2"}


def test_positive_controls_join_the_proxies(tiny_rxrx1):
    adata = mod.add_positive_controls_to_proxies(tiny_rxrx1)
    labels = dict(zip(adata.obs["well_type"], adata.obs["perturbation"], strict=True))
    assert labels["positive_control"] == "control"
    assert labels["negative_control"] == "control"
    assert set(adata.obs.loc[adata.obs["well_type"] == "treatment", "perturbation"]) == {"s1", "s2"}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cross_validate_trains_on_all_other_experiments(tiny_rxrx1, monkeypatch, n):
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    huvec = huvec[~((huvec.obs["batch"] == "HUVEC-3") & (huvec.obs["perturbation"] == "s2"))]
    training_batches = []
    fit_models = mod.fit_models

    def record_fit(train, config):
        training_batches.append(set(train.obs["batch"]))
        return fit_models(train, config)

    monkeypatch.setattr(mod, "fit_models", record_fit)
    folds = list(
        mod.cross_validate(
            huvec, cfg(n_validation_batches=n, bridges=[], positive_controls_as_proxies=True)
        )
    )
    assert len(folds) == len(training_batches) == n
    held_out = [scores["batch"].unique().item() for scores in folds]
    assert len(set(held_out)) == n
    for train, batch in zip(training_batches, held_out, strict=True):
        assert train == {"HUVEC-1", "HUVEC-2", "HUVEC-3"} - {batch}
    assert all("tvn" not in set(scores["model"]) for scores in folds)
    for scores in folds:
        assert (
            scores["n_training_batches"].tolist()
            == scores["perturbation"].map({"s1": 2, "s2": 1}).tolist()
        )


def test_cross_validate_fits_the_neural_bridge(tiny_rxrx1):
    neural = {"device": "cpu", "n_steps": 2, "n_features": 2, "n_features_w": 2}
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    scores = pd.concat(
        mod.cross_validate(huvec, cfg(n_validation_batches=3, bridges=["neural"], neural=neural))
    )
    assert "neural bridge" in set(scores["model"])


def test_nested_cv_tunes_on_inner_validation_and_scores_the_outer_folds(tiny_rxrx1):
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_bridges_rxrx1", ["name=test", "wandb_mode=disabled"])
    config = OmegaConf.merge(
        config,
        {
            "n_validation_batches": 2,
            "inner_folds": 2,
            "n_configs": 2,
            "kernel_grid": {"reg": [1e-3, 1e-2], "lengthscale_scale": [1]},
            "neural_grid": {"reg": [1e-3], "weight_decay": [0.0], "phi_size": [2], "psi_size": [3]},
            "neural_checkpoints": [1, 2],
            "neural": {"device": "cpu", "cross_fit_folds": 0},
        },
    )
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    folds = list(tune.nested_cv(huvec, config))
    assert len(folds) == 2
    for inner, selected, scores in folds:
        assert scores["batch"].nunique() == 1
        assert (scores["n_training_batches"] == 2).all()
        # per inner fold: 2 kernel configurations, 1 neural at 2 checkpoints, the shift reference
        assert inner.groupby("inner_fold").size().tolist() == [2 + 2 + 1] * 2
        assert (inner["bridge"] == "shift (b=1)").sum() == 2
        neural = selected.set_index("bridge").loc["neural"]
        assert neural["n_steps"] in {1, 2}
        assert (neural["phi_size"], neural["psi_size"]) == (2, 3)
        models = set(scores["model"])
        assert {"kernel bridge (tuned)", "neural bridge (tuned)", "shift (b=1)"} <= models
        assert "tvn" not in models


def test_log_summary_writes_every_model_and_metric(monkeypatch):
    logged = {}
    monkeypatch.setattr(mod.wandb, "run", object())
    monkeypatch.setattr(mod.wandb, "summary", logged)
    scores = pd.DataFrame(
        {
            "perturbation": ["s1", "s1"],
            "metric": ["mse", "mse"],
            "model": ["shift", "kernel"],
            "value": [1.0, 3.0],
            "batch": "e1",
        }
    )
    mod.log_summary("fold 2", mod.summary(scores))
    assert logged == {"fold 2/shift/mse": 1.0, "fold 2/kernel/mse": 3.0}


class Spy:
    """Records training and prediction inputs; predicts ones."""

    perturbations_ = np.array(["s1", "s2"])

    def fit(self, adata, *args):
        self.adata = adata
        return self

    def predict(self, perts, test):
        self.test = test
        return np.ones((len(perts), test.n_vars))


def test_standardised_bridge_scales_inputs_and_centres_targets(tiny_rxrx1):
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    original = huvec.X.copy()
    seen = Standardised(Spy()).fit(huvec).model.adata
    ctrl = (seen.obs["perturbation"] == "control").to_numpy()
    np.testing.assert_allclose(seen.X[ctrl].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(seen.X[ctrl].std(axis=0), 1)
    np.testing.assert_allclose(seen.X[~ctrl], huvec.X[~ctrl] - huvec.X[~ctrl].mean(0))
    np.testing.assert_array_equal(huvec.X, original)


def test_prediction_reuses_training_input_scale_and_adds_back_target_mean(tiny_rxrx1):
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    train = huvec[huvec.obs["batch"] != "HUVEC-3"]
    test = huvec[huvec.obs["batch"] == "HUVEC-3"].copy()
    test.X = test.X * 3 + 100
    controls = train.X[(train.obs["perturbation"] == "control").to_numpy()]
    model = Standardised(Spy()).fit(train)
    pred = model.predict(["s1", "s2"], test)
    ctrl = (test.obs["perturbation"] == "control").to_numpy()
    np.testing.assert_allclose(
        model.model.test.X[ctrl], (test.X[ctrl] - controls.mean(0)) / controls.std(0)
    )
    y_mean = train.X[~(train.obs["perturbation"] == "control").to_numpy()].mean(0)
    np.testing.assert_allclose(model.model.test.X[~ctrl], test.X[~ctrl] - y_mean)
    np.testing.assert_allclose(pred, 1 + np.tile(y_mean, (2, 1)))


def test_fit_models_standardises_linear_and_bridge_inputs(tiny_rxrx1):
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    models = mod.fit_models(huvec, cfg(bridges=["kernel"]))
    assert isinstance(models["kernel bridge"], Standardised)
    controls = huvec.X[(huvec.obs["perturbation"] == "control").to_numpy()]
    for name in ["kernel bridge", "shift (b=1)", "pooled (b=0)", "transfer (fit b)"]:
        np.testing.assert_allclose(models[name].w_mean_, controls.mean(0))
        np.testing.assert_allclose(models[name].w_std_, controls.std(0))
    plain = mod.fit_models(huvec, cfg(bridges=["kernel"], standardise_w=False))
    assert plain["kernel bridge"].w_std_ == 1  # targets still centred, inputs left as they are
    np.testing.assert_array_equal(plain["transfer (fit b)"].w_std_, 1)
