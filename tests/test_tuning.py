import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.model.batch_transfer import Standardised

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tune_bridges  # noqa: E402


def test_sample_configs_draws_distinct_grid_points():
    grid = OmegaConf.create({"reg": [1, 2, 3], "scale": [10, 20]})
    drawn = tuning.sample_configs(grid, 4, seed=0)
    every = tuning.sample_configs(grid, 100, seed=0)
    assert len(every) == 6
    assert len(drawn) == 4
    assert all(c in every for c in drawn)
    assert len({tuple(c.items()) for c in drawn}) == 4


def test_selection_ranks_configurations_by_their_mean_over_inner_folds():
    inner = pd.DataFrame(
        {
            "bridge": ["kernel"] * 4 + ["neural"] * 2 + ["shift (b=1)"] * 2,
            "reg": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, None, None],
            "phi_size": [None] * 4 + [8.0, 8.0, None, None],
            "inner_fold": [0, 1, 0, 1, 0, 1, 0, 1],
            "mse": [0.1, 0.9, 0.4, 0.4, 0.3, 0.3, 0.5, 0.7],  # reg 1 wins fold 0, reg 2 on average
            "pearson": [1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 5.0, 7.0],
        }
    )
    averaged = tuning.mean_over_inner_folds(inner)
    assert tuning.select(averaged) == {
        "kernel": {"reg": 2.0},
        "neural": {"reg": 3.0, "phi_size": 8},
    }
    assert isinstance(tuning.select(averaged)["neural"]["phi_size"], int)
    shift = averaged[averaged["bridge"] == "shift (b=1)"].iloc[0]
    assert (shift["mse"], shift["pearson"]) == pytest.approx((0.6, 6.0))


class Spy:
    """Records the data it is fit on; predicts ones."""

    perturbations_ = np.array(["a"])

    def fit(self, adata, *args):
        self.adata = adata
        return self

    def predict(self, perts, test):
        return np.ones((len(perts), test.n_vars))


def test_standardised_leaves_constant_genes_finite():
    obs = pd.DataFrame(
        {"perturbation": ["control"] * 3 + ["a"] * 3, "batch": ["b0", "b1", "b2"] * 2}
    )
    X = np.array([[0.0, 1], [0, 2], [0, 3], [1, 1], [2, 1], [3, 1]])
    model = Standardised(Spy()).fit(ad.AnnData(X, obs=obs))
    assert np.isfinite(model.model.adata.X).all()


def _cells(n_batches=6):
    """Expression of control and perturbations a, b in every batch, with a batch offset."""
    rng = np.random.default_rng(0)
    rows, X = [], []
    for e in range(n_batches):
        offset = rng.normal(size=4)
        for label, mean in [("control", 0.0), ("a", 1.0), ("b", -1.0)]:
            X.append(mean + offset + rng.normal(size=(12, 4)))
            rows += [(label, f"b{e}")] * 12
    obs = pd.DataFrame(rows, columns=["perturbation", "batch"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.vstack(X), obs=obs)


def test_replogle_nested_cv_tunes_on_inner_folds_and_scores_the_outer_folds():
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_bridges", ["name=test"])
    config = OmegaConf.merge(
        config,
        {
            "n_folds": 2,
            "inner_folds": 2,
            "n_configs": 2,
            "kernel_grid": {"reg": [1e-3, 1e-2], "lengthscale_scale": [1]},
            "neural_grid": {"reg": [1e-3], "weight_decay": [0.0], "phi_size": [2], "psi_size": [3]},
            "neural_checkpoints": [1, 2],
            "neural": {"device": "cpu", "cross_fit_folds": 0},
        },
    )
    inner, selected, per_pert, per_batch = tune_bridges.nested_cv(_cells(), config)
    # per outer and inner fold: 2 kernel configurations, 1 neural at 2 checkpoints, shift
    assert inner.groupby(["fold", "inner_fold"]).size().tolist() == [2 + 2 + 1] * 4
    assert set(selected["bridge"]) == {"kernel", "neural"}
    assert {"kernel bridge (tuned)", "neural bridge (tuned)", "shift (b=1)"} <= set(
        per_batch["model"]
    )
