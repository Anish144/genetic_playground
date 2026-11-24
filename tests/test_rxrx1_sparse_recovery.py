import hashlib
import importlib
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
from genetic_perturbation_playground.experiments.checkpoints import read_json

ROOT = Path(__file__).parents[1]


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


def test_virtual_average_rejects_a_missing_reference_batch(sparse, recovery_data):
    controls = recovery_data[
        (recovery_data.obs.perturbation == "control") & (recovery_data.obs.batch != "HUVEC-5")
    ]
    with pytest.raises(ValueError, match="reference.*HUVEC-5"):
        sparse.virtual_average(object(), controls, sparse.reference_pairs(recovery_data))


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


@pytest.fixture
def runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("evaluate_rxrx1_sparse_recovery")


def runner_config(*, variant="per_x", **overrides):
    register_configs()
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(
            "sparse_recovery/rxrx1",
            [
                "name=test",
                f"neural={variant}",
                "proxy_controls=negative",
                "batch_unit=experiment",
                "neural.device=cpu",
            ],
        )
    cfg.bridges = ["kernel", "neural"]
    cfg.include_linear = False
    cfg.n_seeds = 1
    cfg.k_values = [2]
    cfg.n_configs = 1
    cfg.neural_checkpoints = [0, 1]
    cfg.kernel_grid = {"reg": [0.1], "lengthscale_scale": [1.0]}
    cfg.neural_grid = {"reg": [0.1], "psi_size": [2], "width_w": [4]}
    return OmegaConf.merge(cfg, overrides)


def run(runner, data, cfg, directory):
    return runner.run_experiment(
        data, cfg, directory, input_fingerprint=hashlib.sha256(data.X.tobytes()).hexdigest()
    )


def test_config_subsamples_without_changing_existing_tuning_defaults():
    register_configs()
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        original = compose("out_of_batch/rxrx1", ["name=old"])
        cfg = compose("sparse_recovery/rxrx1", ["name=new"])
    assert cfg.k_values == [2, 3, 4, 5]
    assert cfg.n_seeds == 5
    assert cfg.bridges == ["neural"] and cfg.include_linear is True
    assert len(tuning.candidate_configs(cfg)["neural"]) == 20
    assert original.n_configs == 20
    assert original.inner_folds == 3
    assert original.neural.cross_fit_folds == 5
    assert cfg.neural.cross_fit_folds == 0
    assert cfg.neural.seed == cfg.seed
    assert cfg.cell_type == "HUVEC" and cfg.batch_unit == "plate"
    assert cfg.proxy_controls == "positive" and cfg.wandb_mode == "disabled"
    assert cfg.weight_by_cells is True
    assert cfg.kernel_grid == original.kernel_grid
    assert list(cfg.neural_grid.width_w) == [128, 256]
    assert list(cfg.neural_variant_grids.hybrid.deviation_reg) == [1e-2, 1e-1, 1.0]
    assert cfg.neural_checkpoints == [0, 5, 10, 20, 40, 60, 100]
    assert original.neural_checkpoints == [0, 10, 20, 40, 60, 100, 150, 200, 300, 400]


@pytest.mark.parametrize(
    "override",
    [
        {"cell_type": None},
        {"bridges": []},
        {"bridges": ["unknown"]},
        {"bridges": ["neural", "neural"]},
        {"k_values": [2, 2]},
        {"k_values": [1]},
        {"k_values": []},
        {"n_seeds": 0},
        {"n_configs": 0},
        {"seed": -1},
        {"hp_sampling_seed": -1},
        {"hp_sampling_seed": True},
        {"hp_sampling_seed": 1.5},
        {"hp_sampling_seed": "1"},
        {"neural_checkpoints": []},
        {"neural_checkpoints": [-1]},
        {"neural": {"seed": 99}},
    ],
)
def test_runner_rejects_invalid_settings_before_fitting(
    runner, recovery_data, tmp_path, monkeypatch, override
):
    monkeypatch.setattr(tuning, "make_kernel", lambda *args: pytest.fail("fitted invalid config"))
    with pytest.raises(ValueError):
        run(runner, recovery_data, runner_config(**override), tmp_path)


def test_runner_rejects_empty_requested_cohort_before_fitting(
    runner, recovery_data, tmp_path, monkeypatch
):
    data = recovery_data[recovery_data.obs.batch.isin(["HUVEC-0", "HUVEC-1"])]
    monkeypatch.setattr(tuning, "make_kernel", lambda *args: pytest.fail("fitted an empty cohort"))
    with pytest.raises(ValueError, match="eligible|coverage"):
        run(runner, data, runner_config(k_values=[2, 3, 4]), tmp_path)


@pytest.mark.parametrize(
    ("bridge", "variant"),
    [("neural", "per_x"), ("neural", "shared_2nn"), ("neural", "hybrid"), ("kernel", "per_x")],
)
def test_single_bridge_with_linear_models_completes_and_resumes(
    runner, recovery_data, tmp_path, monkeypatch, bridge, variant
):
    cfg = runner_config(variant=variant, bridges=[bridge], include_linear=True)
    omitted = "kernel" if bridge == "neural" else "neural"
    monkeypatch.setattr(
        tuning, f"make_{omitted}", lambda *args: pytest.fail("fitted omitted bridge")
    )
    expected = run(runner, recovery_data, cfg, tmp_path)
    assert set(expected.model) == {
        "retained_mean",
        f"{bridge}_virtual",
        "pooled (b=0)",
        "shift (b=1)",
        "transfer (fit b)",
        "full transfer (fit B)",
    }
    assert np.isfinite(expected.value).all()
    assert set(read_json(tmp_path / "seeds/seed_0/candidates.json")["candidates"]) == {bridge}
    assert set(read_json(tmp_path / "seeds/seed_0/k_2/selected.json")["selected"]) == {bridge}
    assert (tmp_path / "seeds/seed_0/complete.json").exists()
    monkeypatch.setattr(
        tuning, f"make_{bridge}", lambda *args: pytest.fail("refitted completed seed")
    )
    monkeypatch.setattr(
        runner, "_fit_linear", lambda *args: pytest.fail("refitted completed linear models")
    )
    pd.testing.assert_frame_equal(run(runner, recovery_data, cfg, tmp_path), expected)


@pytest.mark.parametrize("grid", ["neural_grid", "neural_variant_grids.hybrid"])
def test_runner_rejects_seed_in_active_tuning_grid(runner, recovery_data, tmp_path, grid):
    register_configs()
    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose(
            "sparse_recovery/rxrx1",
            ["name=test", "seed=3", "neural=hybrid", f"+{grid}.seed=[42]"],
        )
    with pytest.raises(ValueError, match="seed.*grid|grid.*seed"):
        run(runner, recovery_data, cfg, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_hybrid_infinite_linear_penalty_selects_refits_and_resumes(runner, recovery_data, tmp_path):
    from genetic_perturbation_playground.experiments import checkpoints as state

    cfg = runner_config(variant="hybrid", neural={"linear_reg": float("inf")})
    cfg.neural_variant_grids.hybrid.linear_reg = [float("inf")]
    cfg.neural_variant_grids.hybrid.deviation_reg = [None]
    expected = run(runner, recovery_data, cfg, tmp_path)
    selected = state.decode_settings(read_json(tmp_path / "seeds/seed_0/k_2/selected.json"))
    assert np.isposinf(selected["selected"]["neural"]["linear_reg"])
    pd.testing.assert_frame_equal(run(runner, recovery_data, cfg, tmp_path), expected)
    assert np.isfinite(expected.value).all()


def test_source_fingerprint_ignores_unrelated_code_but_tracks_dependencies(
    runner, tmp_path, monkeypatch
):
    for folder in ["src", "scripts"]:
        for source in (ROOT / folder).rglob("*.py"):
            destination = tmp_path / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    original = runner._source_fingerprint()
    for relative in [
        "scripts/rxrx3_core_proxy_minimal_repro.py",
        "src/genetic_perturbation_playground/model/proxy_confounded.py",
    ]:
        path = tmp_path / relative
        path.write_text(path.read_text() + "\n# unrelated change\n")
    assert runner._source_fingerprint() == original
    for relative in [
        "scripts/train_rxrx1_transfer.py",
        "src/genetic_perturbation_playground/model/batch_transfer.py",
        "src/genetic_perturbation_playground/utils/metrics.py",
    ]:
        path = tmp_path / relative
        previous = runner._source_fingerprint()
        path.write_text(path.read_text() + "\n# relevant change\n")
        assert runner._source_fingerprint() != previous


def test_empty_cohorts_save_coverage_and_report_all_missing_k(runner, recovery_data, tmp_path):
    data = recovery_data[recovery_data.obs.batch.isin(["HUVEC-0", "HUVEC-1"])]
    with pytest.raises(ValueError, match=r"k=3.*k=4"):
        run(runner, data, runner_config(k_values=[2, 3, 4]), tmp_path)
    assert (tmp_path / "reference_pairs.csv").exists()
    coverage = pd.read_csv(tmp_path / "eligibility.csv")
    assert coverage.groupby("k").eligible.sum().to_dict() == {2: 2, 3: 0, 4: 0}
    assert set(coverage[~coverage.eligible].reason) == {"fewer_than_k_batches"}


def test_runner_resume_equals_uninterrupted(runner, recovery_data, tmp_path, monkeypatch):
    cfg = runner_config(
        n_seeds=2,
        k_values=[2, 3, 4],
        kernel_grid={"reg": [0.01, 0.1, 1.0]},
        neural_grid={"psi_size": [2, 3, 4]},
    )
    clean = run(runner, recovery_data, cfg, tmp_path / "clean")
    factory = tuning.make_neural
    injected = False

    def interrupt(settings, effective):
        nonlocal injected
        if effective.seed == 1 and not injected:
            injected = True
            raise RuntimeError("stopped during seed 1 tuning")
        return factory(settings, effective)

    monkeypatch.setattr(tuning, "make_neural", interrupt)
    resumed_dir = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="stopped"):
        run(runner, recovery_data, cfg, resumed_dir)
    assert (resumed_dir / "seeds/seed_0/complete.json").exists()
    before = {p: p.stat().st_mtime_ns for p in resumed_dir.glob("seeds/seed_0/**/*.json")}
    resumed = run(runner, recovery_data, cfg, resumed_dir)
    pd.testing.assert_frame_equal(clean, resumed)
    assert before == {p: p.stat().st_mtime_ns for p in before}
    assert not resumed.duplicated(["seed", "k", "perturbation", "model"]).any()
    assert set(resumed.model) == {"retained_mean", "kernel_virtual", "neural_virtual"}
    assert set(resumed.k) == {2, 3, 4} and set(resumed.seed) == {0, 1}
    with np.load(resumed_dir / "seeds/seed_0/k_4/estimates.npz", allow_pickle=False) as saved:
        assert saved["full_mean"].shape == (3, 2)
        c = saved["perturbations"].tolist().index("c")
        np.testing.assert_allclose(saved["full_mean"][c], saved["retained_mean"][c])
    assert len(pd.read_csv(resumed_dir / "summary.csv")) == 12
    for seed in (0, 1):
        directory = resumed_dir / "seeds" / f"seed_{seed}"
        candidates = read_json(directory / "candidates.json")["candidates"]
        trials = list(directory.glob("k_*/trials/fold_*/*/*.json"))
        assert len(trials) == 18
        for trial in trials:
            record = read_json(trial)
            assert record["identity"]["config"] == candidates[trial.parent.name][0]


@pytest.mark.parametrize(
    ("offset", "draw_seeds", "selected_regs"),
    [
        (None, [4, 5], [[6, 8], [5, 7]]),
        (0, [4, 5], [[6, 8], [5, 7]]),
        (7, [11, 12], [[1, 2], [3, 5]]),
    ],
)
def test_each_seed_draws_fresh_candidates_independent_of_launch_range(
    runner, recovery_data, tmp_path, offset, draw_seeds, selected_regs
):
    cfg = runner_config(
        variant="shared_2nn",
        seed=4,
        neural={"seed": 4},
        n_seeds=2,
        k_values=[2, 3],
        n_configs=2,
        hp_sampling_seed=offset,
        kernel_grid={"reg": list(range(1, 9))},
        neural_grid={"reg": list(range(1, 9))},
        neural_variant_grids={"shared_2nn": {"deviation_reg": [0.1]}},
    )
    original = OmegaConf.to_container(cfg, resolve=True)
    combined = tmp_path / "combined"
    run(runner, recovery_data, cfg, combined)
    assert OmegaConf.to_container(cfg, resolve=True) == original
    for seed, draw_seed, regs in zip((4, 5), draw_seeds, selected_regs, strict=True):
        directory = combined / "seeds" / f"seed_{seed}"
        metadata = read_json(directory / "candidates.json")
        assert metadata["hp_sampling_seed"] == draw_seed
        candidates = metadata["candidates"]
        assert set(candidates) == {"kernel", "neural"}
        for values in candidates.values():
            assert [c["reg"] for c in values] == regs
        effective = read_json(directory / "effective_config.json")
        assert effective["seed"] == effective["neural"]["seed"] == seed
        assert effective["hp_sampling_seed"] == draw_seed
        trials = list(directory.glob("k_*/trials/fold_*/*/*.json"))
        assert len(trials) == 20
        for trial in trials:
            record = read_json(trial)["identity"]
            assert record["config"] in candidates[trial.parent.name]
            if trial.parent.name == "neural":
                assert record["model"]["seed"] == seed

    single_cfg = OmegaConf.merge(cfg, {"seed": 5, "neural": {"seed": 5}, "n_seeds": 1})
    run(runner, recovery_data, single_cfg, tmp_path / "single")
    left, right = (tmp_path / name / "seeds/seed_5" for name in ("combined", "single"))
    for name in ("candidates.json", "effective_config.json", "retained_order.csv"):
        assert (left / name).read_bytes() == (right / name).read_bytes()


def test_hp_sampling_seed_preserves_retention_and_initialization(runner, recovery_data, tmp_path):
    cfg = runner_config(variant="hybrid", bridges=["neural"], k_values=[3])
    cfg.neural_grid.reg = [0.1, 0.2, 0.3, 0.4]
    cfg.neural_variant_grids.hybrid.deviation_reg = [0.1]
    cfg.neural_variant_grids.hybrid.linear_reg = [0.01]
    OmegaConf.update(cfg, "hp_sampling_seed", 1, force_add=True)
    candidates = []
    for sampling_seed in (1, 2):
        cfg.hp_sampling_seed = sampling_seed
        directory = tmp_path / f"hp_{sampling_seed}"
        run(runner, recovery_data, cfg, directory)
        identity = read_json(directory / "manifest.json")["identity"]
        candidates.append(read_json(directory / "seeds/seed_0/candidates.json")["candidates"])
        assert identity["settings"]["hp_sampling_seed"] == sampling_seed
        trials = list(directory.glob("seeds/seed_0/k_3/trials/fold_*/neural/*.json"))
        assert len(trials) == 3
        for trial in trials:
            assert read_json(trial)["identity"]["model"]["seed"] == 0

    assert candidates[0] != candidates[1]
    left, right = (tmp_path / f"hp_{seed}" / "seeds/seed_0" for seed in (1, 2))
    assert (left / "retained_order.csv").read_bytes() == (right / "retained_order.csv").read_bytes()
    with (
        np.load(left / "k_3/estimates.npz") as first,
        np.load(right / "k_3/estimates.npz") as second,
    ):
        for field in ("perturbations", "full_mean", "retained_mean"):
            np.testing.assert_array_equal(first[field], second[field])
    with pytest.raises(ValueError, match="incompatible"):
        run(runner, recovery_data, cfg, tmp_path / "hp_1")


def test_completed_seed_resume_and_extension(runner, recovery_data, tmp_path, monkeypatch):
    cfg = runner_config(kernel_grid={"reg": list(range(1, 9))}, n_configs=2)
    expected = run(runner, recovery_data, cfg, tmp_path)
    first_candidates = tmp_path / "seeds/seed_0/candidates.json"
    original_candidates = first_candidates.read_bytes()
    original = tuning.make_kernel
    seeds_fitted = []

    def record(settings, effective):
        assert effective.seed != 0, "completed seed was fitted again"
        seeds_fitted.append(effective.seed)
        return original(settings, effective)

    monkeypatch.setattr(tuning, "make_kernel", record)
    (tmp_path / "scores.csv").unlink()
    (tmp_path / "summary.csv").unlink()
    pd.testing.assert_frame_equal(run(runner, recovery_data, cfg, tmp_path), expected)
    assert seeds_fitted == []
    cfg.n_seeds = 2
    extended = run(runner, recovery_data, cfg, tmp_path)
    assert seeds_fitted == [1] * 5  # two candidates in two folds and one final fit
    assert first_candidates.read_bytes() == original_candidates
    second = read_json(tmp_path / "seeds/seed_1/candidates.json")
    assert [c["reg"] for c in second["candidates"]["kernel"]] == [4, 5]
    assert set(extended.seed) == {0, 1}
    assert (tmp_path / "scores.csv").exists() and (tmp_path / "summary.csv").exists()
    # Lowering the requested seed count must not erase already completed results.
    cfg.n_seeds = 1
    assert set(run(runner, recovery_data, cfg, tmp_path).seed) == {0, 1}


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("corruption", ["candidate", "seed_type"])
def test_resume_rejects_corrupt_seed_candidates(
    runner, recovery_data, tmp_path, monkeypatch, completed, corruption
):
    from genetic_perturbation_playground.experiments.checkpoints import write_json

    cfg = runner_config()
    if completed:
        run(runner, recovery_data, cfg, tmp_path)
    else:

        def stop(*args, **kwargs):
            raise RuntimeError("interrupted tuning")

        monkeypatch.setattr(tuning, "tune_resumable", stop)
        with pytest.raises(RuntimeError, match="interrupted tuning"):
            run(runner, recovery_data, cfg, tmp_path)

    artifact = tmp_path / "seeds/seed_0/candidates.json"
    metadata = read_json(artifact)
    if corruption == "candidate":
        metadata["candidates"]["kernel"][0]["reg"] = 99
    else:
        metadata["hp_sampling_seed"] = False
    write_json(artifact, metadata)
    monkeypatch.setattr(tuning, "make_kernel", lambda *args: pytest.fail("fitted corrupt run"))
    with pytest.raises(ValueError, match="candidates.json"):
        run(runner, recovery_data, cfg, tmp_path)


@pytest.mark.parametrize("completed", [False, True])
def test_reduced_seed_range_checks_candidates_of_all_reported_seeds(
    runner, recovery_data, tmp_path, completed
):
    cfg = runner_config(bridges=["kernel"], n_seeds=2)
    run(runner, recovery_data, cfg, tmp_path)
    directory = tmp_path / "seeds/seed_1"
    if not completed:
        # Simulate stopping after committing k=2 but before committing the seed.
        (directory / "complete.json").unlink()
    (directory / "candidates.json").write_text("{}")
    cfg.n_seeds = 1
    with pytest.raises(ValueError, match="candidates.json"):
        run(runner, recovery_data, cfg, tmp_path)


def test_reordered_grid_dimensions_reject_resume_extension(runner, recovery_data, tmp_path):
    cfg = runner_config(bridges=["kernel"])
    cfg.kernel_grid = {"reg": [0.1, 1.0, 5.0], "lengthscale_scale": [0.2, 0.6, 1.0]}
    run(runner, recovery_data, cfg, tmp_path)
    cfg.kernel_grid = {"lengthscale_scale": [0.2, 0.6, 1.0], "reg": [0.1, 1.0, 5.0]}
    cfg.n_seeds = 2
    with pytest.raises(ValueError, match="incompatible"):
        run(runner, recovery_data, cfg, tmp_path)
    assert not (tmp_path / "seeds/seed_1").exists()


def test_poisoned_removed_outcomes_do_not_change_selection_or_predictions(
    runner, sparse, recovery_data, tmp_path
):
    cfg = runner_config(include_linear=True)
    original = tmp_path / "original"
    altered = tmp_path / "altered"
    run(runner, recovery_data, cfg, original)
    order = sparse.retained_order(sparse.reference_pairs(recovery_data), cfg.seed)
    kept = sparse.retained_data(recovery_data, order, 2)
    poisoned = recovery_data.copy()
    poisoned.X[~poisoned.obs_names.isin(kept.obs_names)] += 1000
    run(runner, poisoned, cfg, altered)
    left, right = original / "seeds/seed_0/k_2", altered / "seeds/seed_0/k_2"
    assert read_json(left / "selected.json") == read_json(right / "selected.json")
    with np.load(left / "estimates.npz") as a, np.load(right / "estimates.npz") as b:
        for key in [
            "retained_mean",
            "kernel_virtual",
            "neural_virtual",
            "pooled (b=0)",
            "shift (b=1)",
            "transfer (fit b)",
            "full transfer (fit B)",
        ]:
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0)
        assert not np.allclose(a["full_mean"], b["full_mean"])


def test_negative_only_proxies_never_train_on_positive_controls(
    runner, recovery_data, tmp_path, monkeypatch
):
    extra = recovery_data[recovery_data.obs.perturbation == "a"].copy()
    extra.obs["perturbation"] = "positive"
    extra.obs["well_type"] = "positive_control"
    extra.X[:] = 1e9
    data = ad.concat([recovery_data, extra], index_unique="-")
    original = tuning.make_kernel

    def record(settings, cfg):
        model = original(settings, cfg)
        fit = model.fit

        def check(train):
            assert set(train.obs.well_type) == {"treatment", "negative_control"}
            assert "positive" not in set(train.obs.perturbation)
            return fit(train)

        model.fit = check
        return model

    monkeypatch.setattr(tuning, "make_kernel", record)
    scores = run(runner, data, runner_config(), tmp_path)
    assert set(scores.perturbation) == {"a", "b", "c"}


def test_changed_inputs_and_corrupt_completed_results_are_rejected(
    runner, recovery_data, tmp_path, monkeypatch
):
    cfg = runner_config()
    run(runner, recovery_data, cfg, tmp_path)
    monkeypatch.setattr(tuning, "make_kernel", lambda *args: pytest.fail("reused incompatible run"))
    changed = recovery_data.copy()
    changed.X[0, 0] += 1
    with pytest.raises(ValueError, match="incompatible"):
        run(runner, changed, cfg, tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        run(runner, recovery_data, runner_config(n_configs=2), tmp_path)
    artifact = tmp_path / "seeds/seed_0/k_2/scores.csv"
    artifact.write_text("broken")
    with pytest.raises(ValueError, match="scores.csv"):
        run(runner, recovery_data, cfg, tmp_path)


def test_final_fit_failure_keeps_completed_tuning(runner, recovery_data, tmp_path, monkeypatch):
    original = tuning.fit_selected
    monkeypatch.setattr(
        tuning,
        "fit_selected",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("refit stopped")),
    )
    with pytest.raises(RuntimeError, match="refit stopped"):
        run(runner, recovery_data, runner_config(), tmp_path)
    paths = list(tmp_path.glob("seeds/seed_0/k_2/trials/fold_*/*/*.json"))
    before = {p: p.read_bytes() for p in paths}
    assert len(paths) == 4
    monkeypatch.setattr(tuning, "fit_selected", original)
    run(runner, recovery_data, runner_config(), tmp_path)
    assert before == {p: p.read_bytes() for p in paths}


def test_summaries_use_equal_seed_weights_and_handle_zero_baseline(sparse):
    rows = []
    for seed, pairs in [(0, [("a", 4, 1), ("b", 0, 0)]), (1, [("a", 10, 5)])]:
        for perturbation, baseline, virtual in pairs:
            for model, value in [
                ("retained_mean", baseline),
                ("kernel_virtual", virtual),
                ("neural_virtual", virtual),
            ]:
                rows.append(
                    dict(
                        seed=seed,
                        k=2,
                        perturbation=perturbation,
                        model=model,
                        value=value,
                        n_reference_batches=2 if perturbation == "b" else 4,
                        n_retained_batches=2,
                        n_removed_batches=0 if perturbation == "b" else 2,
                        metric="mse",
                    )
                )
    per_seed, overall = sparse.summarize_scores(pd.DataFrame(rows))
    result = overall[(overall.model == "kernel_virtual") & (overall.cohort == "all")].iloc[0]
    assert result.baseline_mse == 6.0  # mean of seed means: 2 and 10
    assert result.virtual_mse == 2.75  # mean of 0.5 and 5
    assert result.improvement == 3.25
    assert result.win_fraction == 0.75  # tie does not count as improvement
    assert result.n_seeds == 2
    zero_rows = pd.DataFrame(rows)
    zero_rows = zero_rows[zero_rows.perturbation == "b"]
    empty, _ = sparse.summarize_scores(zero_rows)
    assert (empty[empty.cohort == "removed_only"].n_perturbations == 0).all()
    assert empty[empty.cohort == "removed_only"].virtual_mse.isna().all()
