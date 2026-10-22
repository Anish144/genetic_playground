import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from genetic_perturbation_playground.utils.metrics import control_whitener

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_batch_transfer_rxrx1 as mod  # noqa: E402


def cfg(**overrides):
    return OmegaConf.merge(
        OmegaConf.load(ROOT / "configs" / "train_batch_transfer_rxrx1.yaml"), overrides
    )


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
