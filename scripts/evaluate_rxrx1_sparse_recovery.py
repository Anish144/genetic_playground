"""Compare virtual and sparse observed averages with the full RxRx1 mean.

Run or resume with the same name/output_dir:

    uv run python scripts/evaluate_rxrx1_sparse_recovery.py name=huvec_sparse n_seeds=10

Each seed retains k response batches per perturbation for k in [2,3,4].
Controls remain available everywhere. Tuning uses only retained responses;
the eight sampled configurations per bridge are reused across seeds/k, while
fitting, validation and selection are repeated for each seed/k. Completed
trials and seeds are reused from the stable output directory. An interrupted
neural trial restarts; its optimizer state is not saved.
"""

import subprocess
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from train_rxrx1_transfer import select_batch_unit, select_cell_type, select_proxy_controls

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.config import register_configs
from genetic_perturbation_playground.data.rxrx1 import load_rxrx1, rxrx1_path
from genetic_perturbation_playground.data.transfer import is_control
from genetic_perturbation_playground.experiments import checkpoints as state
from genetic_perturbation_playground.experiments import rxrx1_sparse_recovery as recovery

register_configs()
ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_SETTINGS = {"name", "output_dir", "wandb_mode", "resume", "n_seeds"}


def _validate_config(cfg):
    for key in ("n_seeds", "n_configs"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if cfg.neural.seed != cfg.seed:
        raise ValueError(
            "Set seed to control both retention and neural initialization; neural.seed must match."
        )
    if not isinstance(cfg.cell_type, str) or not cfg.cell_type:
        raise ValueError(
            "cell_type must name a single cell type; pooled cell_type=null is unsupported."
        )
    if list(cfg.bridges) != ["kernel", "neural"]:
        raise ValueError("This experiment tunes both bridges; use bridges=[kernel,neural].")
    ks = list(cfg.k_values)
    if (
        not ks
        or any(type(k) is not int or k not in (2, 3, 4) for k in ks)
        or len(set(ks)) != len(ks)
    ):
        raise ValueError("k_values must be a nonempty, distinct subset of [2,3,4].")
    steps = list(cfg.neural_checkpoints)
    if (
        not steps
        or any(type(s) is not int or s < 0 for s in steps)
        or len(set(steps)) != len(steps)
    ):
        raise ValueError("neural_checkpoints must be distinct nonnegative integers.")


def _scientific_settings(cfg):
    return {
        key: value
        for key, value in OmegaConf.to_container(cfg, resolve=True).items()
        if key not in OPERATIONAL_SETTINGS
    }


def _source_fingerprint():
    """Conservatively invalidate resume after any package or script edit."""
    paths = sorted((ROOT / "src/genetic_perturbation_playground").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    return state.fingerprint_settings(
        {str(p.relative_to(ROOT)): state.fingerprint_file(p) for p in paths}
    )


def _provenance(cfg):
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return {
        "input_path": str(rxrx1_path(**cfg.embeddings).resolve()),
        "git_revision": git.stdout.strip() if git.returncode == 0 else None,
    }


def _write_csv(path, table):
    content = table.to_csv(index=False).encode()
    state.atomic_write(path, lambda stream: stream.write(content))


def _save_seed_inputs(directory, order, cfg):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "retained_order.csv"
    content = order.to_csv(index=False).encode()
    if path.exists() and path.read_bytes() != content:
        raise ValueError(
            f"Retained order mismatch at {path}; restore it or use a fresh output directory."
        )
    if not path.exists():
        state.atomic_write(path, lambda stream: stream.write(content))
    path = directory / "effective_config.json"
    expected = _scientific_settings(cfg)
    if path.exists() and state.read_json(path) != expected:
        raise ValueError(
            f"Effective configuration mismatch at {path}; use a fresh output directory."
        )
    if not path.exists():
        state.write_json(path, expected)


def _prepare_data(data, cfg):
    data = select_cell_type(data, cfg.cell_type)
    data = select_batch_unit(data, cfg.batch_unit)
    pairs = recovery.reference_pairs(data)
    if pairs.empty:
        raise ValueError("No treatment reference pairs are available.")
    coverage = recovery.eligibility_table(pairs, list(cfg.k_values))
    for k in cfg.k_values:
        if not coverage[coverage.k == k].eligible.any():
            maximum = int(coverage.n_reference_batches.max())
            raise ValueError(
                f"No eligible perturbations for k={k}; maximum observed coverage is {maximum} batches. Check the embeddings or k_values."
            )
    data = select_proxy_controls(data, cfg)
    data = data[is_control(data) | data.obs.well_type.eq("treatment").to_numpy()]
    return data, pairs, coverage


def _save_references(directory, pairs, coverage, run_id):
    marker = directory / "reference_complete.json"
    if state.is_complete(marker, identity={"run_id": run_id}):
        return
    _write_csv(directory / "reference_pairs.csv", pairs)
    _write_csv(directory / "eligibility.csv", coverage)
    state.write_completion(
        marker,
        [directory / "reference_pairs.csv", directory / "eligibility.csv"],
        identity={"run_id": run_id},
    )


def _stage_identity(run_id, seed, k=None):
    return {"run_id": run_id, "seed": seed} | ({"k": k} if k is not None else {})


def _print_workload(cfg, candidates):
    folds = cfg.n_seeds * sum(cfg.k_values)
    trials = {bridge: folds * len(configs) for bridge, configs in candidates.items()}
    print(
        f"Planned workload: {trials['kernel']} kernel tuning fits, {trials['neural']} neural tuning runs, "
        f"{cfg.n_seeds * len(cfg.k_values)} final refits per bridge; "
        f"{trials['neural'] * max(cfg.neural_checkpoints)} neural tuning updates. "
        "Completed work will be skipped.",
        flush=True,
    )


def run_experiment(
    data: ad.AnnData,
    cfg: DictConfig,
    output_dir: Path,
    *,
    input_fingerprint: str,
) -> pd.DataFrame:
    """Run or resume sparse-response recovery in one stable output directory.

    Parameters
    ----------
    data : anndata.AnnData
        Unmasked RxRx1 embeddings and original metadata. Input is not mutated.
    cfg : omegaconf.DictConfig
        Experiment preset, including the base seed and requested seed count.
    output_dir : pathlib.Path
        Persistent result directory; one writer is permitted at a time.
    input_fingerprint : str
        Content fingerprint of the complete input file, including metadata,
        computed by the caller. This function does not verify the supplied hash.

    Returns
    -------
    pandas.DataFrame
        MSE rows from all validated completed seed/k stages, including earlier
        seeds if n_seeds is reduced. Increasing n_seeds appends new seeds.

    Raises
    ------
    ValueError
        For invalid settings, insufficient coverage, incompatible resume or
        damaged committed artifacts. Existing results are never overwritten
        by an incompatible run.
    """
    _validate_config(cfg)
    candidates = tuning.candidate_configs(cfg)
    identity = {
        "settings": _scientific_settings(cfg),
        "input_fingerprint": input_fingerprint,
        "source_fingerprint": _source_fingerprint(),
        "candidates": candidates,
    }
    run_id = state.fingerprint_settings(identity)
    with state.run_lock(output_dir):
        state.ensure_manifest(
            output_dir,
            {"schema_version": 1, "identity": identity, "provenance": _provenance(cfg)},
            resume=cfg.resume,
        )
        data, pairs, coverage = _prepare_data(data, cfg)
        _save_references(output_dir, pairs, coverage, run_id)
        _print_workload(cfg, candidates)
        for seed in range(cfg.seed, cfg.seed + cfg.n_seeds):
            directory = output_dir / "seeds" / f"seed_{seed}"
            marker = directory / "complete.json"
            pair_markers = [directory / f"k_{k}" / "complete.json" for k in cfg.k_values]
            if state.is_complete(marker, identity=_stage_identity(run_id, seed)):
                for k, pair_marker in zip(cfg.k_values, pair_markers, strict=True):
                    if not state.is_complete(
                        pair_marker, identity=_stage_identity(run_id, seed, k)
                    ):
                        raise ValueError(
                            f"Missing completed stage {pair_marker}; restore the checkpoint."
                        )
                print(f"skip completed seed {seed}", flush=True)
                continue
            effective = OmegaConf.merge(cfg, {"seed": seed, "neural": {"seed": seed}})
            order = recovery.retained_order(pairs, seed)
            _save_seed_inputs(directory, order, effective)
            for k in cfg.k_values:
                _run_pair(
                    data,
                    pairs,
                    coverage,
                    order,
                    effective,
                    candidates,
                    directory / f"k_{k}",
                    k,
                    run_id,
                )
                _rebuild_results(output_dir, cfg, run_id)
            state.write_completion(
                marker,
                [
                    directory / "retained_order.csv",
                    directory / "effective_config.json",
                    *pair_markers,
                ],
                identity=_stage_identity(run_id, seed),
            )
            print(f"completed seed {seed}", flush=True)
        return _rebuild_results(output_dir, cfg, run_id)


def _run_pair(data, pairs, coverage, order, cfg, candidates, directory, k, run_id):
    identity = _stage_identity(run_id, cfg.seed, k)
    if state.is_complete(directory / "complete.json", identity=identity):
        print(f"skip completed seed {cfg.seed}, k={k}", flush=True)
        return
    print(f"seed {cfg.seed}, k={k}: tuning retained responses", flush=True)
    inner, selected, winners = tuning.tune_resumable(
        recovery.inner_folds(data, order, k),
        cfg,
        directory=directory / "trials",
        candidates=candidates,
    )
    _write_csv(directory / "inner_scores.csv", inner)
    state.write_json(
        directory / "selected.json",
        {"selected": selected, "validation": winners.to_dict(orient="index")},
    )
    train = recovery.retained_data(data, order, k)
    models = tuning.fit_selected(train, selected, cfg, f"seed {cfg.seed}/k {k}/neural bridge")
    eligible = coverage[(coverage.k == k) & coverage.eligible]
    labels = eligible.perturbation.tolist()
    reference = pairs[pairs.perturbation.isin(labels)]
    controls = data[is_control(data)]
    full = recovery.observed_average(data, labels)
    retained = recovery.observed_average(train, labels)
    virtual = {
        f"{bridge}_virtual": recovery.virtual_average(
            models[f"{bridge} bridge (tuned)"], controls, reference
        ).loc[labels]
        for bridge in ("kernel", "neural")
    }
    scores = recovery.score_estimates(full, retained, virtual, eligible, seed=cfg.seed, k=k)
    per_seed, _ = recovery.summarize_scores(scores)
    arrays = {
        "perturbations": np.asarray(labels, dtype=str),
        "features": np.asarray(full.columns, dtype=str),
        "full_mean": full.to_numpy(),
        "retained_mean": retained.to_numpy(),
        **{name: frame.to_numpy() for name, frame in virtual.items()},
    }
    state.atomic_write(
        directory / "estimates.npz", lambda stream: np.savez_compressed(stream, **arrays)
    )
    _write_csv(directory / "scores.csv", scores)
    _write_csv(directory / "summary.csv", per_seed)
    artifacts = [
        directory / name
        for name in [
            "inner_scores.csv",
            "selected.json",
            "estimates.npz",
            "scores.csv",
            "summary.csv",
        ]
    ]
    state.write_completion(directory / "complete.json", artifacts, identity=identity)
    print(per_seed.to_string(index=False), flush=True)


def _rebuild_results(directory, cfg, run_id):
    tables = []
    for seed_dir in sorted((directory / "seeds").glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for k in cfg.k_values:
            stage = seed_dir / f"k_{k}"
            if state.is_complete(
                stage / "complete.json", identity=_stage_identity(run_id, seed, k)
            ):
                tables.append(pd.read_csv(stage / "scores.csv", dtype={"perturbation": str}))
    if not tables:
        return pd.DataFrame()
    scores = pd.concat(tables, ignore_index=True).sort_values(
        ["seed", "k", "perturbation", "model"], ignore_index=True
    )
    if scores.duplicated(["seed", "k", "perturbation", "model"]).any():
        raise ValueError("Duplicate completed experiment scores.")
    _, summary = recovery.summarize_scores(scores)
    _write_csv(directory / "scores.csv", scores)
    _write_csv(directory / "summary.csv", summary)
    return scores


@hydra.main(
    config_path="../configs", config_name="evaluate_rxrx1_sparse_recovery", version_base=None
)
def main(cfg: DictConfig):
    """Load RxRx1, execute the requested seed range, and report saved results."""
    _validate_config(cfg)
    path = rxrx1_path(**cfg.embeddings)
    fingerprint = state.fingerprint_file(path)
    data = load_rxrx1(**cfg.embeddings)
    directory = Path(to_absolute_path(cfg.output_dir))
    with wandb.init(
        project="genetic-playground",
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb_mode,
    ):
        scores = run_experiment(data, cfg, directory, input_fingerprint=fingerprint)
        if wandb.run is not None:
            wandb.summary.update(
                {
                    f"k {k}/{model}/mse": value
                    for (k, model), value in scores.groupby(["k", "model"]).value.mean().items()
                }
            )
    print(f"Saved {len(scores)} score rows to {directory}", flush=True)


if __name__ == "__main__":
    main()
