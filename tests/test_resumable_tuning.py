from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.config import register_configs
from genetic_perturbation_playground.experiments.checkpoints import read_json, write_json

ROOT = Path(__file__).parents[1]


def tiny_config(variant="per_x"):
    register_configs()
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(
            "tune_rxrx1_transfer",
            ["name=test", f"neural={variant}", "neural.device=cpu", "neural.cross_fit_folds=0"],
        )
    cfg.n_configs = 1
    cfg.neural_checkpoints = [0, 1, 2]
    cfg.kernel_grid = {"reg": [0.1], "lengthscale_scale": [1.0]}
    cfg.neural_grid = {"reg": [0.1], "psi_size": [2], "width_w": [4]}
    OmegaConf.update(
        cfg,
        "neural_variant_grids",
        {
            "shared_2nn": {"deviation_reg": [None]},
            "hybrid": {"deviation_reg": [None], "linear_reg": [0.0]},
        },
        merge=False,
    )
    return cfg


def evaluator(data):
    reference = data[(data.obs.perturbation == "control") & (data.obs.batch == "b0")]

    def evaluate(model):
        predicted = model.predict_environment_average(["a"], reference)
        return {"mse": float(np.mean((predicted - [[11, 23]]) ** 2))}

    return evaluate


@pytest.mark.parametrize("variant", ["per_x", "shared_2nn", "hybrid"])
def test_completed_tuning_resume_never_refits_and_preserves_variant(
    virtual_train, tmp_path, monkeypatch, variant
):
    cfg = tiny_config(variant)
    candidates = tuning.candidate_configs(cfg)
    folds = [(virtual_train, evaluator(virtual_train))] * 2
    expected = tuning.tune_resumable(folds, cfg, directory=tmp_path, candidates=candidates)
    assert len(expected[0]) == 8
    assert len(list(tmp_path.glob("fold_*/*/*.json"))) == 4
    assert not expected[0].duplicated(["inner_fold", "bridge", "candidate_id", "n_steps"]).any()
    if variant != "per_x":
        assert expected[1]["neural"]["deviation_reg"] is None
    if variant == "hybrid":
        assert expected[1]["neural"]["linear_reg"] == 0.0

    def forbidden(*args):
        raise AssertionError("A completed trial was fitted again")

    monkeypatch.setattr(tuning, "make_kernel", forbidden)
    monkeypatch.setattr(tuning, "make_neural", forbidden)
    actual = tuning.tune_resumable(folds, cfg, directory=tmp_path, candidates=candidates)
    pd.testing.assert_frame_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_interrupted_trial_restarts_and_matches_uninterrupted_run(
    virtual_train, tmp_path, monkeypatch
):
    cfg = tiny_config()
    candidates = tuning.candidate_configs(cfg)
    evaluate = evaluator(virtual_train)
    calls = 0

    def interrupt(model):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("interrupted")
        return evaluate(model)

    directory = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="interrupted"):
        tuning.tune_resumable(
            [(virtual_train, interrupt)] * 2, cfg, directory=directory, candidates=candidates
        )
    assert len(list(directory.glob("fold_*/*/*.partial.json"))) == 1
    expected = tuning.tune_resumable(
        [(virtual_train, evaluate)] * 2, cfg, directory=tmp_path / "clean", candidates=candidates
    )
    factories = {bridge: getattr(tuning, f"make_{bridge}") for bridge in ["kernel", "neural"]}
    fits = []
    for bridge, original in factories.items():

        def record(settings, cfg, bridge=bridge, original=original):
            fits.append(bridge)
            return original(settings, cfg)

        monkeypatch.setattr(tuning, f"make_{bridge}", record)
    resumed = tuning.tune_resumable(
        [(virtual_train, evaluate)] * 2, cfg, directory=directory, candidates=candidates
    )
    assert fits == ["neural"]
    pd.testing.assert_frame_equal(resumed[0], expected[0])
    assert resumed[1] == expected[1]
    assert not list(directory.glob("fold_*/*/*.partial.json"))


def test_selection_uses_all_folds_and_stable_ties(virtual_train, tmp_path):
    cfg = tiny_config()
    cfg.n_configs = 2
    cfg.kernel_grid.reg = [0.2, 0.1]
    candidates = tuning.candidate_configs(cfg)
    _, selected, _ = tuning.tune_resumable(
        [(virtual_train, lambda model: {"mse": 1.0})] * 2,
        cfg,
        directory=tmp_path,
        candidates=candidates,
    )
    assert selected["kernel"]["reg"] == 0.2
    assert selected["neural"]["n_steps"] == 0


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonfinite"])
def test_partial_or_nonfinite_trials_cannot_select_a_winner(virtual_train, tmp_path, damage):
    cfg = tiny_config()
    candidates = tuning.candidate_configs(cfg)
    folds = [(virtual_train, evaluator(virtual_train))]
    tuning.tune_resumable(folds, cfg, directory=tmp_path, candidates=candidates)
    path = next(tmp_path.glob("fold_*/neural/*.json"))
    record = read_json(path)
    if damage == "missing":
        record["rows"].pop()
    elif damage == "duplicate":
        record["rows"].append(record["rows"][0])
    else:
        path.write_text('{"mse": NaN}')
    if damage != "nonfinite":
        write_json(path, record)
    with pytest.raises(ValueError, match="checkpoint|trial"):
        tuning.tune_resumable(folds, cfg, directory=tmp_path, candidates=candidates)


def test_nonfinite_new_scores_are_not_committed(virtual_train, tmp_path):
    cfg = tiny_config()
    with pytest.raises(ValueError, match="finite"):
        tuning.tune_resumable(
            [(virtual_train, lambda model: {"mse": np.nan})],
            cfg,
            directory=tmp_path,
            candidates=tuning.candidate_configs(cfg),
        )
    assert not list(tmp_path.glob("fold_*/*/*.json"))


def test_candidates_subsample_both_grids_without_replacement():
    register_configs()
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose("tune_rxrx1_transfer", ["name=test", "n_configs=8"])
    candidates = tuning.candidate_configs(cfg)
    assert candidates == tuning.candidate_configs(cfg)
    for values in candidates.values():
        assert len(values) == 8
        assert len({tuple(sorted(v.items())) for v in values}) == 8
    small = tiny_config()
    small.n_configs = 8
    assert all(len(values) == 1 for values in tuning.candidate_configs(small).values())


def test_omitting_kernel_preserves_the_ten_sampled_neural_candidates():
    register_configs()
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose("evaluate_rxrx1_sparse_recovery", ["name=test", "n_configs=10"])
    original = tuning.candidate_configs(cfg)["neural"]
    cfg.bridges = ["neural"]
    candidates = tuning.candidate_configs(cfg)
    assert set(candidates) == {"neural"}
    assert len(candidates["neural"]) == 10
    assert candidates["neural"] == original
