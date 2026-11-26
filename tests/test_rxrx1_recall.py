from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from genetic_perturbation_playground.config import register_configs
from genetic_perturbation_playground.experiments.rxrx1_recall import (
    gene_mapping,
    gene_profiles,
    huvec_estimates,
    known_pairs,
    recall_table,
    tail_recall,
)


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


def tiny_config():
    register_configs()
    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base=None):
        overrides = ["neural.device=cpu", "neural.n_steps=1", "neural.cross_fit_folds=0"]
        return compose("recall/rxrx1", ["name=test", *overrides])


def test_observed_averages_each_sirnas_own_plates():
    estimates = huvec_estimates(tiny_huvec(), tiny_config())
    assert list(estimates) == ["observed", "bridge"]
    assert list(estimates["observed"].index) == ["a", "b"]
    np.testing.assert_allclose(estimates["observed"].loc["a"], [10.65, 9.5])  # plates p0, p1


def test_bridge_predicts_every_sirna_from_every_plate():
    estimates = huvec_estimates(tiny_huvec(), tiny_config())
    bridge = estimates["bridge"]
    assert bridge.index.equals(estimates["observed"].index)
    assert list(bridge.columns) == ["f0", "f1"]
    assert np.isfinite(bridge.to_numpy()).all()
