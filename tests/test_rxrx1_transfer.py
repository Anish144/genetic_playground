import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from genetic_perturbation_playground.model.batch_transfer import BatchTransferModel
from genetic_perturbation_playground.utils.metrics import control_whitener

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


def test_whitened_mse_of_perfect_prediction_is_zero(tiny_rxrx1):
    test = tiny_rxrx1[tiny_rxrx1.obs["batch"] == "HUVEC-1"]
    obs_means = {
        p: test.X[(test.obs["perturbation"] == p).to_numpy()].mean(0) for p in ["s1", "s2"]
    }

    class Oracle:
        perturbations_ = np.array(["s1", "s2"])

        def predict(self, perts, test):
            return np.stack([obs_means[p] for p in perts])

    scores = mod.score_batch({"oracle": Oracle()}, test, lambda z: 2 * z)
    np.testing.assert_allclose(scores["value"], 0.0, atol=1e-12)
    assert set(scores["metric"]) == {"mse", "mse_raw"}


def test_positive_controls_are_never_scored(tiny_rxrx1):
    test = tiny_rxrx1[tiny_rxrx1.obs["batch"] == "HUVEC-1"]

    class Zero:
        perturbations_ = np.array(["p1", "s1", "s2"])

        def predict(self, perts, test):
            return np.zeros((len(perts), test.n_vars))

    scores = mod.score_batch({"zero": Zero()}, test, lambda z: z)
    assert set(scores["perturbation"]) == {"s1", "s2"}


def test_positive_controls_join_the_proxies(tiny_rxrx1):
    adata = mod.add_positive_controls_to_proxies(tiny_rxrx1)
    labels = dict(zip(adata.obs["well_type"], adata.obs["perturbation"], strict=True))
    assert labels["positive_control"] == "control"
    assert labels["negative_control"] == "control"
    assert set(adata.obs.loc[adata.obs["well_type"] == "treatment", "perturbation"]) == {"s1", "s2"}


def test_whitener_sees_training_negative_controls_only(tiny_rxrx1, monkeypatch):
    seen = []

    def recording_whitener(X, batch, eps):
        seen.append((set(batch), len(X)))
        return control_whitener(X, batch, eps)

    monkeypatch.setattr(mod, "control_whitener", recording_whitener)
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    folds = list(
        mod.cross_validate(huvec, cfg(n_folds=3, bridges=[], positive_controls_as_proxies=True))
    )
    assert len(folds) == 3
    for fold, (scores, (batches, n_crops)) in enumerate(zip(folds, seen, strict=True)):
        assert set(scores["fold"]) == {fold}
        assert "tvn" in set(scores["model"])
        assert not batches & set(scores["batch"])
        assert n_crops == 2 * 30


def test_cross_validate_fits_the_neural_bridge(tiny_rxrx1):
    neural = {"device": "cpu", "n_steps": 2, "n_features": 2, "n_features_w": 2}
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    scores = pd.concat(mod.cross_validate(huvec, cfg(n_folds=3, bridges=["neural"], neural=neural)))
    assert "neural bridge" in set(scores["model"])


def test_nested_cv_tunes_on_inner_validation_and_scores_the_outer_folds(tiny_rxrx1):
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_bridges_rxrx1", ["name=test", "wandb_mode=disabled"])
    config = OmegaConf.merge(
        config,
        {
            "n_folds": 3,
            "n_configs": 2,
            "kernel_grid": {"reg": [1e-3, 1e-2], "lengthscale_scale": [1]},
            "neural_grid": {"reg": [1e-3], "weight_decay": [0.0], "phi_size": [2], "psi_size": [3]},
            "neural_checkpoints": [1, 2],
            "neural": {"device": "cpu", "cross_fit_folds": 0},
        },
    )
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    folds = list(tune.nested_cv(huvec, config))
    assert len(folds) == 3
    for inner, selected, scores in folds:
        assert len(inner) == 2 + 2  # 2 kernel configurations, 1 neural at 2 checkpoints
        neural = selected.set_index("bridge").loc["neural"]
        assert neural["n_steps"] in {1, 2}
        assert (neural["phi_size"], neural["psi_size"]) == (2, 3)
        models = set(scores["model"])
        assert {"kernel bridge (tuned)", "neural bridge (tuned)", "tvn", "shift (b=1)"} <= models


def test_log_summary_writes_every_model_and_metric(monkeypatch):
    logged = {}
    monkeypatch.setattr(mod.wandb, "run", object())
    monkeypatch.setattr(mod.wandb, "summary", logged)
    scores = pd.DataFrame(
        {
            "perturbation": ["s1", "s1", "s1", "s1"],
            "metric": ["mse", "mse_raw", "mse", "mse_raw"],
            "model": ["shift", "shift", "tvn", "tvn"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "batch": "e1",
        }
    )
    mod.log_summary("fold 2", mod.summary(scores))
    assert logged == {
        "fold 2/shift/mse": 1.0,
        "fold 2/shift/mse_raw": 2.0,
        "fold 2/tvn/mse": 3.0,
        "fold 2/tvn/mse_raw": 4.0,
    }


def test_standardised_shift_predicts_as_shift(tiny_rxrx1):
    """shift is unchanged by a per-dimension affine map, so standardising must be undone exactly."""
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    is_test = (huvec.obs["batch"] == "HUVEC-3").to_numpy()
    train, test = huvec[~is_test], huvec[is_test]
    plain = BatchTransferModel(slope=1.0).fit(train)
    standardised = mod.Standardised(BatchTransferModel(slope=1.0)).fit(train)
    perts = ["s1", "s2"]
    np.testing.assert_allclose(standardised.predict(perts, test), plain.predict(perts, test))


def test_standardised_uses_negative_control_statistics(tiny_rxrx1):
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    model = mod.Standardised(BatchTransferModel(slope=1.0)).fit(huvec)
    negative = huvec[(huvec.obs["well_type"] == "negative_control").to_numpy()]
    batch = negative.obs["batch"].to_numpy()
    centred = negative.X - pd.DataFrame(negative.X).groupby(batch).transform("mean").to_numpy()
    np.testing.assert_allclose(model.mean_, negative.X.mean(axis=0))
    np.testing.assert_allclose(model.std_, centred.std(axis=0))


def test_fit_models_standardises_only_the_bridges(tiny_rxrx1):
    huvec = mod.select_cell_type(tiny_rxrx1, "HUVEC")
    models = mod.fit_models(huvec, cfg(bridges=["kernel"], standardise_bridges=True))
    assert isinstance(models["kernel bridge"], mod.Standardised)
    assert not isinstance(models["shift (b=1)"], mod.Standardised)
    plain = mod.fit_models(huvec, cfg(bridges=["kernel"], standardise_bridges=False))
    assert not isinstance(plain["kernel bridge"], mod.Standardised)
