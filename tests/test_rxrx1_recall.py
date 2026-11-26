from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from genetic_perturbation_playground.config import register_configs, resolve_model_config
from genetic_perturbation_playground.experiments.rxrx1_recall import (
    claim_output,
    fit_bridge,
    fit_linear,
    gene_mapping,
    gene_profiles,
    huvec_estimates,
    known_pairs,
    recall_table,
    tail_recall,
    tune_bridge,
    tune_linear,
    virtual_folds,
)
from genetic_perturbation_playground.experiments.rxrx1_sparse_recovery import (
    observed_average,
    reference_pairs,
    retained_order,
    virtual_average,
)
from genetic_perturbation_playground.model.batch_transfer import build_neural
from genetic_perturbation_playground.model.linear_transfer import LinearTransfer


def related_profiles():
    """40 random gene profiles where g0 ≈ g1 (top tail) and g2 = -g3 (bottom tail)."""
    values = np.random.default_rng(0).normal(size=(40, 50))
    values[1] = values[0] + 0.01
    values[3] = -values[2]
    return pd.DataFrame(values, index=[f"g{i}" for i in range(40)])


def test_gene_profiles_maps_averages_and_centres():
    means = pd.DataFrame(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [5.0, 5.0]], index=["s1", "s2", "s3", "s9"]
    )
    mapping = pd.Series({"s1": "A", "s2": "A", "s3": "B"})
    profiles, n_unmapped = gene_profiles(means, mapping)
    assert n_unmapped == 1
    assert list(profiles.index) == ["A", "B"]
    np.testing.assert_allclose(profiles.to_numpy(), [[1.0, -1.0], [-1.0, 1.0]])


def test_gene_mapping_repairs_spreadsheet_symbols_and_excludes_ambiguous():
    symbols = pd.Series(
        {"s1": "LSM\xa07.00", "s2": "CLP 1", "s3": "TIMM23B;TIMM23", "s4": None, "s5": "C9orf72"}
    )
    mapping, excluded = gene_mapping(symbols)
    assert mapping.to_dict() == {"s1": "LSM7", "s2": "CLP1", "s5": "C9orf72"}
    assert excluded.to_dict() == {"s3": "multiple genes", "s4": "no gene symbol"}


def test_gene_mapping_rejects_unrecognised_symbols():
    with pytest.raises(ValueError, match="s1"):
        gene_mapping(pd.Series({"s1": "LSM 7.5"}))


def test_known_pairs_unordered_and_deduplicated():
    table = pd.DataFrame({"entity1": ["A", "B", "A", "A"], "entity2": ["B", "A", "A", "Z"]})
    assert known_pairs(table, pd.Index(["A", "B", "C"])) == {("A", "B")}


def test_tail_recall_counts_both_tails():
    pairs = {("g0", "g1"), ("g2", "g3")}
    assert tail_recall(related_profiles(), pairs, tail=0.05) == (2, 1.0)


def test_tail_recall_misses_pairs_outside_the_tails():
    profiles = related_profiles()
    g4, g5 = profiles.loc["g4"], profiles.loc["g5"]
    profiles.loc["g5"] = g5 - (g5 @ g4) / (g4 @ g4) * g4  # cosine(g4, g5) = 0: mid-distribution
    assert tail_recall(profiles, {("g4", "g5")}, tail=0.05) == (0, 0.0)


def test_tail_recall_rejects_nonfinite_cosines():
    profiles = related_profiles()
    profiles.loc["g9"] = 0.0  # zero norm: undefined cosine
    with pytest.raises(ValueError, match="finite"):
        tail_recall(profiles, {("g0", "g1")})


def test_recall_table_names_a_database_without_pairs():
    means = related_profiles()
    mapping = pd.Series(means.index, index=means.index)
    empty = pd.DataFrame({"entity1": ["X"], "entity2": ["Y"]})
    with pytest.raises(ValueError, match="SIGNOR: no known pairs"):
        recall_table({"observed": means}, mapping, {"SIGNOR": empty})


def test_recall_table_reports_counts():
    profiles = related_profiles()
    means = profiles.set_axis([f"s{i}" for i in range(40)])
    mapping = pd.Series(profiles.index, index=means.index)
    database = pd.DataFrame({"entity1": ["g0", "g2"], "entity2": ["g1", "g3"]})
    table = recall_table({"observed": means, "bridge": means}, mapping, {"DB": database})
    assert list(table.columns) == [
        "estimator",
        "database",
        "n_genes",
        "n_unmapped",
        "n_pairs",
        "n_hits",
        "recall",
    ]
    assert table.estimator.tolist() == ["observed", "bridge"]
    assert table.n_pairs.tolist() == [2, 2]
    assert table.n_genes.tolist() == [40, 40]
    assert table.recall.between(0, 1).all()


def tiny_huvec():
    """Three plates with positive-control proxies; siRNA "a" is missing from plate p2."""
    rows, values = [], []
    for plate in range(3):
        for label, well_type, offset in [
            ("control", "positive_control", 0.0),
            ("a", "treatment", 10.0),
            ("b", "treatment", 20.0),
        ]:
            if label == "a" and plate == 2:
                continue
            for crop in range(4):
                rows.append((label, f"p{plate}", well_type))
                values.append([offset + plate + 0.1 * crop, offset - plate])
    obs = pd.DataFrame(rows, columns=["perturbation", "batch", "well_type"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.array(values), obs=obs, var=pd.DataFrame(index=["f0", "f1"]))


def tiny_config(extra=()):
    register_configs()
    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base=None):
        overrides = ["neural.device=cpu", "neural.n_steps=1", "neural.cross_fit_folds=0"]
        return compose("recall/rxrx1", ["name=test", *overrides, *extra])


def fitted_models(data):
    cfg = tiny_config()
    neural = build_neural(resolve_model_config(cfg, "neural"), standardise_w=cfg.standardise_w)
    return {
        "shift (b=1)": LinearTransfer(slope=1.0, standardise_w=False).fit(data),
        "neural bridge": neural.fit(data),
    }


def test_observed_averages_each_sirnas_own_plates():
    estimates = huvec_estimates(tiny_huvec(), {})
    assert list(estimates) == ["observed"]
    assert list(estimates["observed"].index) == ["a", "b"]
    np.testing.assert_allclose(estimates["observed"].loc["a"], [10.65, 9.5])  # plates p0, p1


def test_models_average_every_sirna_over_every_plate():
    data = tiny_huvec()
    estimates = huvec_estimates(data, fitted_models(data))
    assert list(estimates) == ["observed", "shift (b=1)", "neural bridge"]
    for name in ["shift (b=1)", "neural bridge"]:
        assert estimates[name].index.equals(estimates["observed"].index)
        assert list(estimates[name].columns) == ["f0", "f1"]
        assert np.isfinite(estimates[name].to_numpy()).all()
    # shift predicts "a" on p2 too: effect (10, 10) plus each plate's control mean
    np.testing.assert_allclose(estimates["shift (b=1)"].loc["a"], [11.15, 9.0])


def test_recall_config_uses_the_hybrid_bridge():
    assert tiny_config().neural.variant == "hybrid"


def additive_huvec(n_plates=4, n_extra=0):
    """Plates holding controls, "a", "b" and `n_extra` more siRNAs.

    Crop = effect + plate offset + 0.1 * crop, so shift (b=1) fits it exactly.
    """
    effects = {"control": (0.0, 0.0), "a": (10.0, 0.0), "b": (0.0, 10.0)}
    effects |= {f"s{i}": (float(i), -float(i)) for i in range(n_extra)}
    rows, values = [], []
    for plate in range(n_plates):
        for label, (e0, e1) in effects.items():
            well_type = "positive_control" if label == "control" else "treatment"
            for crop in range(4):
                rows.append((label, f"p{plate}", well_type))
                values.append([e0 + plate + 0.1 * crop, e1 - plate + 0.1 * crop])
    obs = pd.DataFrame(rows, columns=["perturbation", "batch", "well_type"])
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.array(values), obs=obs, var=pd.DataFrame(index=["f0", "f1"]))


def pairs(data):
    """The (siRNA, plate) pairs of the treatment wells."""
    obs = data.obs[data.obs.well_type.eq("treatment")]
    return set(zip(obs.perturbation, obs.batch, strict=True))


def test_virtual_folds_hide_each_pair_once_and_keep_controls():
    data = additive_huvec()
    folds = virtual_folds(data, n_folds=2, seed=0)
    assert len(folds) == 2
    hidden = []
    for train, _ in folds:
        controls = train.obs.batch[train.obs.well_type.eq("positive_control")]
        assert set(controls) == set(data.obs.batch)
        hidden += sorted(pairs(data) - pairs(train))
    assert sorted(hidden) == sorted(pairs(data))


def test_virtual_fold_score_is_zero_for_an_exact_model():
    for train, evaluate in virtual_folds(additive_huvec(), n_folds=2, seed=0):
        shift = LinearTransfer(slope=1.0, standardise_w=False).fit(train)
        assert evaluate(shift)["mse"] == pytest.approx(0.0, abs=1e-10)


def test_virtual_fold_score_penalises_ignoring_plates():
    train, evaluate = virtual_folds(additive_huvec(), n_folds=2, seed=0)[0]
    pooled = LinearTransfer(slope=0.0, standardise_w=False).fit(train)
    assert evaluate(pooled)["mse"] > 0.1


def test_virtual_folds_reject_sirnas_with_too_few_plates():
    data = additive_huvec()
    data = data[~(data.obs.perturbation.eq("a") & data.obs.batch.ne("p0"))]
    with pytest.raises(ValueError, match="'a'"):
        virtual_folds(data, n_folds=2, seed=0)


def test_tune_linear_scores_pooled_and_shift_as_references():
    folds = virtual_folds(additive_huvec(), n_folds=2, seed=0)
    _, scores = tune_linear(folds, [1e-6], standardise_w=False)
    shift = scores[scores.model.eq("shift (b=1)")]
    assert len(shift) == 2 and shift.reg.isna().all()
    assert shift.mse.max() == pytest.approx(0.0, abs=1e-10)
    assert scores[scores.model.eq("pooled (b=0)")].mse.min() > 0.1


def test_tune_linear_picks_a_reg_per_fitted_model():
    folds = virtual_folds(additive_huvec(), n_folds=2, seed=0)
    selected, scores = tune_linear(folds, [1e-6, 1e2], standardise_w=False)
    assert selected == {"transfer (fit b)": 1e-6, "full transfer (fit B)": 1e-6}
    assert set(scores.columns) == {"model", "reg", "fold", "mse"}
    fitted = scores[scores.reg.notna()]
    assert len(fitted) == 2 * 2 * 2  # models x regs x folds
    assert fitted.groupby("reg").mse.mean()[1e2] > 0.1  # a heavy penalty shrinks the true slope


def test_tune_linear_warns_at_the_grid_edge(capsys):
    folds = virtual_folds(additive_huvec(), n_folds=2, seed=0)
    tune_linear(folds, [1e-6, 1e2], standardise_w=False)
    assert "edge of the grid" in capsys.readouterr().out


def test_fit_linear_uses_each_models_reg():
    regs = {"transfer (fit b)": 1e-6, "full transfer (fit B)": 1e-2}
    models = fit_linear(additive_huvec(), regs, standardise_w=False)
    assert list(models) == [
        "pooled (b=0)",
        "shift (b=1)",
        "transfer (fit b)",
        "full transfer (fit B)",
    ]
    assert models["transfer (fit b)"].reg == 1e-6
    assert models["full transfer (fit B)"].reg == 1e-2


def tune_tiny_bridge(directory, extra=()):
    cfg = tiny_config(["n_configs=1", "inner_folds=2", "neural_checkpoints=[0,1]", *extra])
    data = additive_huvec()
    return tune_bridge(virtual_folds(data, 2, cfg.seed), cfg, directory)


def test_tune_bridge_returns_settings_without_a_final_fit(tmp_path):
    settings, scores = tune_tiny_bridge(tmp_path)
    assert {"psi_size", "n_steps", "linear_reg"} <= set(settings)
    assert set(scores.bridge) == {"neural"}
    assert set(scores.inner_fold) == {0, 1}


def test_tune_bridge_resumes_at_the_first_unfinished_trial(tmp_path, capsys):
    settings, scores = tune_tiny_bridge(tmp_path)
    trials = sorted(tmp_path.glob("fold_*/neural/*.json"))
    assert len(trials) == 2
    trials[1].unlink()  # as if the job died during the second fold
    capsys.readouterr()
    resumed, rescored = tune_tiny_bridge(tmp_path)
    out = capsys.readouterr().out
    assert out.count("skip completed trial") == 1
    assert "fold 1 checkpoint" in out and "fold 0 checkpoint" not in out
    assert resumed == settings
    assert (
        rescored.loc[rescored.inner_fold.eq(0), "mse"].tolist()
        == scores.loc[scores.inner_fold.eq(0), "mse"].tolist()
    )


def test_claim_output_resumes_the_same_run_and_refuses_another_config(tmp_path):
    data = additive_huvec()
    claim_output(data, tiny_config(), tmp_path)
    claim_output(data, tiny_config(), tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        claim_output(data, tiny_config(["seed=1"]), tmp_path)


def test_claim_output_refuses_changed_embeddings_on_the_same_plates(tmp_path):
    data = additive_huvec()
    claim_output(data, tiny_config(), tmp_path)
    changed = data.copy()
    changed.X = changed.X + 1.0
    with pytest.raises(ValueError, match="incompatible"):
        claim_output(changed, tiny_config(), tmp_path)


def test_fit_bridge_with_tuned_or_fixed_settings():
    cfg = tiny_config()
    data = additive_huvec()
    tuned = {"lr": 1e-3, "reg": 1e-3, "weight_decay": 0.0, "width_w": 8, "psi_size": 4}
    for settings in [tuned | {"n_steps": 1}, {}]:
        model = fit_bridge(data, cfg, settings)
        assert np.isfinite(huvec_estimates(data, {"bridge": model})["bridge"].to_numpy()).all()


class CountingModel:
    """Delegate to a fitted model, counting environment-average predictions."""

    def __init__(self, model):
        self.model, self.calls = model, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict_environment_average(self, *args, **kwargs):
        self.calls += 1
        return self.model.predict_environment_average(*args, **kwargs)


def test_virtual_fold_score_predicts_each_plate_once():
    data = additive_huvec(n_plates=6, n_extra=18)
    train, evaluate = virtual_folds(data, n_folds=2, seed=0)[0]
    model = CountingModel(LinearTransfer(slope=1.0, standardise_w=False).fit(train))
    assert evaluate(model)["mse"] == pytest.approx(0.0, abs=1e-10)
    assert model.calls <= 6


def test_virtual_fold_score_matches_the_virtual_average():
    data = additive_huvec(n_plates=6, n_extra=18)
    controls = data[data.obs.well_type.eq("positive_control")]
    order = retained_order(reference_pairs(data), seed=0)
    hidden = order[order["rank"] % 2 == 0]
    train, evaluate = virtual_folds(data, n_folds=2, seed=0)[0]
    model = LinearTransfer(slope=0.5, standardise_w=False).fit(train)
    keys = pd.MultiIndex.from_arrays([data.obs.perturbation, data.obs.batch])
    in_hidden = keys.isin(pd.MultiIndex.from_frame(hidden[["perturbation", "batch"]]))
    observed = observed_average(data[in_hidden], sorted(hidden.perturbation.unique()))
    expected = ((virtual_average(model, controls, hidden) - observed) ** 2).to_numpy().mean()
    assert evaluate(model)["mse"] == pytest.approx(expected)
