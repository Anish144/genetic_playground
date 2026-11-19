import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.config import register_configs
from genetic_perturbation_playground.utils.standardisation import (
    Standardised,
)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_replogle_transfer as expression  # noqa: E402
import tune_replogle_transfer  # noqa: E402

register_configs()


def test_validation_batches_are_distinct_and_reproducible():
    batch = np.array(["b0", "b0", "b1", "b2", "b3"])
    selected = tuning.validation_batches(batch, 2, np.random.default_rng(4))
    assert len(selected) == len(set(selected)) == 2
    assert set(selected) <= {"b0", "b1", "b2", "b3"}
    np.testing.assert_array_equal(
        selected, tuning.validation_batches(batch[::-1], 2, np.random.default_rng(4))
    )
    assert set(tuning.validation_batches(batch, 4, np.random.default_rng(4))) == {
        "b0",
        "b1",
        "b2",
        "b3",
    }


@pytest.mark.parametrize("batch,n", [([], 1), (["b0"], 1), (["b0", "b1"], 0), (["b0", "b1"], 3)])
def test_validation_batches_reject_invalid_counts(batch, n):
    with pytest.raises(ValueError):
        tuning.validation_batches(np.array(batch), n, np.random.default_rng(0))


def test_sample_configs_draws_distinct_grid_points():
    grid = OmegaConf.create({"reg": [1, 2, 3], "scale": [10, 20]})
    drawn = tuning.sample_configs(grid, 4, seed=0)
    every = tuning.sample_configs(grid, 100, seed=0)
    assert len(every) == 6
    assert len(drawn) == 4
    assert all(c in every for c in drawn)
    assert len({tuple(c.items()) for c in drawn}) == 4


def test_per_x_grid_varies_hidden_width_independently_of_output_dimension():
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_replogle_transfer_perx", ["name=test"])
    drawn = tuning.sample_configs(config.neural_grid, config.n_configs, config.seed)
    assert len(drawn) == config.n_configs
    for settings in drawn:
        model = tuning.make_neural(settings, config).model
        assert model.bridge_kwargs["width_w"] == settings["width_w"]
        assert model.bridge_kwargs["n_features_w"] == settings["psi_size"]


def test_selection_ranks_configurations_by_their_mean_over_inner_folds():
    inner = pd.DataFrame(
        {
            "bridge": ["kernel"] * 4 + ["neural"] * 2 + ["shift (b=1)"] * 2,
            "reg": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, None, None],
            "psi_size": [None] * 4 + [8.0, 8.0, None, None],
            "width_w": [None] * 4 + [128.0, 128.0, None, None],
            "inner_fold": [0, 1, 0, 1, 0, 1, 0, 1],
            "n_steps": [None] * 4 + [1, 1, None, None],
            "mse": [0.1, 0.9, 0.4, 0.4, 0.3, 0.3, 0.5, 0.7],  # reg 1 wins fold 0, reg 2 on average
            "pearson": [1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 5.0, 7.0],
        }
    )
    averaged = tuning.mean_over_inner_folds(inner)
    config = OmegaConf.create(
        {
            "kernel_grid": {"reg": [1, 2]},
            "neural_grid": {"reg": [3], "psi_size": [8], "width_w": [128]},
        }
    )
    selected = tuning.select(averaged, config)
    assert selected == {
        "kernel": {"reg": 2.0},
        "neural": {"reg": 3.0, "psi_size": 8, "width_w": 128, "n_steps": 1},
    }
    assert isinstance(selected["neural"]["psi_size"], int)
    assert isinstance(selected["neural"]["width_w"], int)
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
    """Controls and a in every batch; b in the first four batches, with a batch offset."""
    rng = np.random.default_rng(0)
    rows, X = [], []
    for e in range(n_batches):
        offset = rng.normal(size=4)
        for label, mean in [("control", 0.0), ("a", 1.0), ("b", -1.0)]:
            if label == "b" and e >= 4:
                continue
            X.append(mean + offset + rng.normal(size=(12, 4)))
            rows += [(label, f"b{e}")] * 12
    obs = pd.DataFrame(rows, columns=["perturbation", "batch"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.vstack(X), obs=obs)


@pytest.mark.parametrize("n", [1, 2])
def test_replogle_validation_trains_on_all_other_batches(monkeypatch, n):
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("train_replogle_transfer", ["name=test"])
    config.n_validation_batches, config.bridges = n, []
    training_batches = []
    fit = expression.fit

    def record_fit(train, cfg):
        training_batches.append(set(train.obs["batch"]))
        return fit(train, cfg)

    monkeypatch.setattr(expression, "fit", record_fit)
    per_pert, per_batch, per_pair = expression.cross_validate(_cells(), config)
    held_out = per_batch["batch"].unique()
    assert len(training_batches) == len(held_out) == n
    for train, batch in zip(training_batches, held_out, strict=True):
        assert train == {"b0", "b1", "b2", "b3", "b4", "b5"} - {batch}
    assert not per_pert.empty
    assert per_pair["batch"].nunique() == n
    assert set(per_pair["metric"]) == {"mse", "pearson", "pearson_top100"}
    assert not per_pair.duplicated(["perturbation", "batch", "model", "metric"]).any()
    assert per_pert.groupby("perturbation")["n_training_batches"].unique().map(list).to_dict() == {
        "a": [5],
        "b": [3],
    }


def test_replogle_nested_cv_tunes_on_inner_folds_and_scores_selected_batches():
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_replogle_transfer", ["name=test", "+neural_grid.width_w=[4]"])
    config = OmegaConf.merge(
        config,
        {
            "n_validation_batches": 2,
            "inner_folds": 2,
            "n_configs": 2,
            "kernel_grid": {"reg": [1e-3, 1e-2], "lengthscale_scale": [1]},
            "neural_grid": {"reg": [1e-3], "weight_decay": [0.0], "psi_size": [3]},
            "neural_checkpoints": [1, 2],
            "neural": {"device": "cpu", "cross_fit_folds": 0},
        },
    )
    inner, selected, per_pert, per_batch, per_pair = tune_replogle_transfer.nested_cv(
        _cells(), config
    )
    assert per_batch["batch"].nunique() == 2
    assert per_pair.groupby("fold")["batch"].nunique().tolist() == [1, 1]
    for scores in [per_pert, per_pair]:
        assert scores.groupby("perturbation")["n_training_batches"].unique().map(
            list
        ).to_dict() == {"a": [5], "b": [3]}
    # per outer and inner fold: 2 kernel configurations, 1 neural at 2 checkpoints, shift
    assert inner.groupby(["fold", "inner_fold"]).size().tolist() == [2 + 2 + 1] * 4
    assert set(selected["bridge"]) == {"kernel", "neural"}
    assert {"kernel bridge (tuned)", "neural bridge (tuned)", "shift (b=1)"} <= set(
        per_batch["model"]
    )
    # one row per (perturbation, held-out batch, model); averaging over perturbations gives per_batch
    assert set(per_pair["metric"]) == {"mse", "pearson", "pearson_top100"}
    assert not per_pair.duplicated(["perturbation", "batch", "model", "metric"]).any()
    from_pairs = (
        per_pair.query("metric == 'mse'").groupby(["batch", "model"])["value"].mean().sort_index()
    )
    batch_mse = (
        per_batch.query("metric == 'mse'").set_index(["batch", "model"])["value"].sort_index()
    )
    pd.testing.assert_series_equal(from_pairs, batch_mse, check_names=False)


def test_trial_settings_override_presets_without_mutation():
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_replogle_transfer", ["name=test", "neural=hybrid"])
    before = OmegaConf.to_container(config, resolve=True)
    first = tuning.make_neural(
        {"psi_size": 3, "width_w": 4, "n_features_w": 2, "seed": 19, "log_key": "first"}, config
    ).model
    second = tuning.make_neural({"psi_size": 6, "log_key": "second"}, config).model
    assert first.variant == second.variant == "hybrid"
    assert (
        first.bridge_kwargs["width_w"],
        first.bridge_kwargs["n_features_w"],
        first.bridge_kwargs["seed"],
    ) == (4, 2, 19)
    assert (second.bridge_kwargs["width_w"], second.bridge_kwargs["n_features_w"]) == (6, 6)
    assert first.bridge_kwargs["log_key"] == "first"
    assert second.bridge_kwargs["log_key"] == "second"
    assert OmegaConf.to_container(config, resolve=True) == before


def test_selected_null_deviation_overrides_numeric_preset():
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose(
            "tune_replogle_transfer",
            [
                "name=test",
                "neural=shared_2nn",
                "neural.deviation_reg=0.2",
                "neural.device=cpu",
                "neural.cross_fit_folds=0",
            ],
        )
    config.kernel_grid = {"reg": [0.001], "lengthscale_scale": [2]}
    config.neural_grid = {"reg": [0.005], "deviation_reg": [None, 0.2], "psi_size": [2]}
    inner = pd.DataFrame(
        [
            {"bridge": "kernel", "reg": 0.001, "lengthscale_scale": 2, "mse": 0.3, "inner_fold": 0},
            {
                "bridge": "neural",
                "reg": 0.005,
                "deviation_reg": None,
                "psi_size": 2,
                "n_steps": 1,
                "mse": 0.1,
                "inner_fold": 0,
            },
            {
                "bridge": "neural",
                "reg": 0.005,
                "deviation_reg": 0.2,
                "psi_size": 2,
                "n_steps": 2,
                "mse": 0.9,
                "inner_fold": 0,
            },
        ]
    )
    selected = tuning.select(tuning.mean_over_inner_folds(inner), config)
    assert selected["neural"] == {"reg": 0.005, "deviation_reg": None, "psi_size": 2, "n_steps": 1}
    assert selected["kernel"] == {"reg": 0.001, "lengthscale_scale": 2}
    model = tuning.fit_selected(_cells(), selected, config)["neural bridge (tuned)"].model.bridge_
    assert model.deviation_reg == model.reg == 0.005
    assert model.n_steps == 1
    assert config.neural.deviation_reg == 0.2


def test_selection_restores_integer_model_settings():
    config = OmegaConf.create(
        {
            "kernel_grid": {"reg": [0.01]},
            "neural_grid": {
                "reg": [0.01],
                "psi_size": [2],
                "n_features_w": [2],
                "seed": [19],
                "gene_block_size": [8],
            },
        }
    )
    rows = pd.DataFrame(
        [
            {"bridge": "kernel", "reg": 0.01, "mse": 0.2},
            {
                "bridge": "neural",
                "reg": 0.01,
                "psi_size": 2.0,
                "n_features_w": 2.0,
                "seed": 19.0,
                "gene_block_size": 8.0,
                "n_steps": 1.0,
                "mse": 0.1,
            },
        ]
    )
    selected = tuning.select(rows, config)["neural"]
    for name, value in [
        ("psi_size", 2),
        ("n_features_w", 2),
        ("seed", 19),
        ("gene_block_size", 8),
        ("n_steps", 1),
    ]:
        assert selected[name] == value
        assert type(selected[name]) is int


@pytest.mark.parametrize("extra", [{"weight_by_cells": False}, {"device": "cpu"}])
def test_mixed_type_grid_winners_can_be_refit(extra):
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        config = compose("tune_replogle_transfer", ["name=test"])
    config.kernel_grid = {"reg": [0.001], "lengthscale_scale": [1]}
    config.neural_grid = {"reg": [0.005], "psi_size": [2]} | {
        key: [value] for key, value in extra.items()
    }
    rows = pd.DataFrame(
        [
            {"bridge": "kernel", "reg": 0.001, "lengthscale_scale": 1, "mse": 0.3, "inner_fold": 0},
            {
                "bridge": "neural",
                "reg": 0.005,
                "psi_size": 2,
                "n_steps": 1,
                "mse": 0.1,
                "inner_fold": 0,
                **extra,
            },
        ]
    )
    selected = tuning.select(tuning.mean_over_inner_folds(rows), config)
    model = tuning.make_neural(selected["neural"], config).model
    assert tuning.make_kernel(selected["kernel"], config).model.reg == 0.001
    assert model.bridge_kwargs["reg"] == 0.005
    assert type(selected["neural"]["reg"]) is float
    if "weight_by_cells" in extra:
        assert model.weight_by_cells is False
    else:
        assert model.bridge_kwargs["device"] == "cpu"
