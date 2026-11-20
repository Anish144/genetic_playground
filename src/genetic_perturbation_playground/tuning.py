"""Batch selection and random-search tuning of the kernel and neural bridges.

Shared by scripts/tune_replogle_transfer.py (Replogle) and scripts/tune_rxrx1_transfer.py. cfg.n_configs
configurations are drawn from each bridge's grid, the same draw in every fold. On each inner fold,
shift (b=1, a reference only), every kernel configuration and every neural configuration at each
of cfg.neural_checkpoints are fit on the other inner folds and scored on this one. The
configuration with the lowest validation mse averaged over the inner folds is selected.
"""

import itertools
from collections.abc import Callable, Collection, Iterable
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig

from genetic_perturbation_playground.config import resolve_model_config
from genetic_perturbation_playground.experiments.checkpoints import (
    fingerprint_settings,
    read_json,
    write_json,
)
from genetic_perturbation_playground.model.batch_transfer import (
    build_kernel,
    build_neural,
)
from genetic_perturbation_playground.model.linear_transfer import (
    LinearTransfer,
)

INT_SETTINGS = [
    "psi_size",
    "width_w",
    "n_features_w",
    "n_steps",
    "depth",
    "cross_fit_folds",
    "seed",
    "gene_block_size",
]
Evaluate = Callable[[object], dict]  # fitted model -> its validation scores, {metric: value}


def _metrics(columns) -> list[str]:
    """The metric columns among `columns`: mse, pearson and pearson_top<k> for any k."""
    return [c for c in columns if c in ("mse", "pearson") or c.startswith("pearson_top")]


def batch_folds(batch: np.ndarray, n_folds: int, rng: np.random.Generator) -> list:
    """The distinct batches of `batch`, shuffled and split into n_folds groups."""
    return np.array_split(rng.permutation(np.unique(batch)), n_folds)


def validation_batches(batch: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Select n distinct batches without replacement for leave-one-batch-out evaluation.

    Each selected batch is held out in a separate run; all other batches train that run,
    including batches selected for other runs. Requires at least two batches and
    1 <= n <= the number of distinct batches. The supplied RNG controls the selection.
    """
    batches = np.unique(batch)
    if len(batches) < 2:
        raise ValueError("leave-one-batch-out evaluation requires at least two batches")
    if not 1 <= n <= len(batches):
        raise ValueError(f"n_validation_batches must be between 1 and {len(batches)}, got {n}")
    return rng.permutation(batches)[:n]


def sample_configs(grid: DictConfig, n: int, seed: int) -> list[dict]:
    """n distinct configurations drawn at random from the grid (all of them if n covers it)."""
    configs = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]
    picks = np.random.default_rng(seed).choice(len(configs), min(n, len(configs)), replace=False)
    return [configs[i] for i in sorted(picks)]


def make_kernel(config: dict, cfg: DictConfig):
    """Unfitted kernel bridge with one configuration: reg and lengthscale_scale."""
    settings = resolve_model_config(cfg, "kernel", config)
    return build_kernel(settings, standardise_w=cfg.standardise_w)


def make_neural(config: dict, cfg: DictConfig):
    """Unfitted neural bridge with one configuration; the rest of its settings from cfg.neural."""
    settings = resolve_model_config(cfg, "neural", _network_sizes(config))
    return build_neural(settings, standardise_w=cfg.standardise_w)


def _network_sizes(config: dict) -> dict:
    """Expand psi_size into control width and output dimension; explicit settings win."""
    settings = dict(config)
    psi = settings.pop("psi_size")
    return {"width_w": psi, "n_features_w": psi} | settings


def tune(folds: Iterable[tuple[sc.AnnData, Evaluate]], cfg: DictConfig) -> tuple:
    """Tune both bridges on the inner folds, each given as (training data, its Evaluate).

    Returns the validation scores on every inner fold, the selected configuration of each bridge,
    and the mean validation scores of the selected configurations and of shift.
    """
    inner = _inner_scores(folds, cfg)
    averaged = mean_over_inner_folds(inner)
    selected, winners = select(averaged, cfg), _winners(averaged)
    print(f"selected: {selected}\n{winners.to_string(float_format='%.4g')}", flush=True)
    return inner, selected, winners


def _inner_scores(folds: Iterable[tuple[sc.AnnData, Evaluate]], cfg: DictConfig) -> pd.DataFrame:
    """Validation scores of shift and every configuration on each inner fold."""
    tables = []
    for j, (train, evaluate) in enumerate(folds):
        print(f"  inner fold {j + 1}/{cfg.inner_folds}", flush=True)
        tables.append(pd.DataFrame(_validation_rows(train, evaluate, cfg)).assign(inner_fold=j))
    return pd.concat(tables, ignore_index=True)


def _validation_rows(train: sc.AnnData, evaluate: Evaluate, cfg: DictConfig) -> list[dict]:
    """Validation scores of shift, each kernel configuration and each neural one at each checkpoint."""
    return (
        _shift_rows(train, evaluate, cfg)
        + _kernel_rows(train, evaluate, cfg)
        + _neural_rows(train, evaluate, cfg)
    )


def _shift_rows(train: sc.AnnData, evaluate: Evaluate, cfg: DictConfig) -> list[dict]:
    """Validation scores of shift (b=1), the reference the bridges are compared with."""
    config = replace(resolve_model_config(cfg, "linear"), slope=1.0)
    scores = evaluate(LinearTransfer(**asdict(config)).fit(train))
    print(f"  shift (b=1): mse {scores['mse']:.4g}", flush=True)
    return [{"bridge": "shift (b=1)", **scores}]


def _kernel_rows(train: sc.AnnData, evaluate: Evaluate, cfg: DictConfig) -> list[dict]:
    """Validation scores of each sampled kernel configuration."""
    rows = []
    for config in sample_configs(cfg.kernel_grid, cfg.n_configs, cfg.seed):
        scores = evaluate(make_kernel(config, cfg).fit(train))
        rows.append({"bridge": "kernel", **config, **scores})
        print(f"  kernel {config}: mse {scores['mse']:.4g}", flush=True)
    return rows


def _neural_rows(train: sc.AnnData, evaluate: Evaluate, cfg: DictConfig) -> list[dict]:
    """Validation scores of each sampled neural configuration at every checkpoint."""
    checkpoints = list(cfg.neural_checkpoints)
    rows = []
    for config in sample_configs(_neural_grid(cfg), cfg.n_configs, cfg.seed):
        model = make_neural(config | {"n_steps": max(checkpoints)}, cfg)

        def record(step, model=model, config=config):
            scores = evaluate(model)
            rows.append({"bridge": "neural", **config, "n_steps": step, **scores})
            print(f"  neural {config} step {step}: mse {scores['mse']:.4g}", flush=True)

        model.fit(train, checkpoints, record)
    return rows


def _neural_grid(cfg: DictConfig) -> dict:
    """Add this variant's grid; explicit neural_grid settings take precedence."""
    variants = cfg.get("neural_variant_grids", {})
    extra = variants.get(cfg.neural.variant, {}) if variants else {}
    return dict(extra) | dict(cfg.neural_grid)


def mean_over_inner_folds(inner: pd.DataFrame) -> pd.DataFrame:
    """Each configuration's (and shift's) validation scores averaged over the inner folds."""
    metrics = _metrics(inner.columns)
    settings = [c for c in inner.columns if c not in [*metrics, "inner_fold"]]
    return inner.groupby(settings, dropna=False)[metrics].mean().reset_index()


def select(averaged: pd.DataFrame, cfg: DictConfig) -> dict:
    """The configuration of each bridge with the lowest mean validation mse."""
    settings = {
        "kernel": list(cfg.kernel_grid),
        "neural": list(dict.fromkeys([*_neural_grid(cfg), "n_steps"])),
    }
    return {
        bridge: _best(averaged[averaged["bridge"] == bridge], names)
        for bridge, names in settings.items()
    }


def _winners(averaged: pd.DataFrame) -> pd.DataFrame:
    """Mean validation scores of the best configuration of each bridge, and of shift."""
    best = averaged.loc[averaged.groupby("bridge")["mse"].idxmin()].set_index("bridge")
    return best[_metrics(best.columns)]


def _best(rows: pd.DataFrame, setting_names: Collection[str]) -> dict:
    """Keep this bridge's winning settings, including explicit null penalties."""
    best = rows.loc[rows["mse"].idxmin(), list(setting_names)]
    settings = {}
    for key, value in best.items():
        if isinstance(value, np.generic):
            value = value.item()  # mixed-type pandas rows retain NumPy scalars
        if pd.isna(value):
            value = None
        elif key in INT_SETTINGS:
            value = int(value)
        settings[key] = value
    return settings


def fit_selected(train: sc.AnnData, selected: dict, cfg: DictConfig, log_key=None) -> dict:
    """Both bridges in their selected configurations, fit on `train`; the neural bridge logs its
    loss to W&B under `log_key` if one is given."""
    kernel = make_kernel(selected["kernel"], cfg)
    neural = make_neural(selected["neural"] | {"log_key": log_key}, cfg)
    return {"kernel bridge (tuned)": kernel.fit(train), "neural bridge (tuned)": neural.fit(train)}


def candidate_configs(cfg: DictConfig) -> dict[str, list[dict]]:
    """Sample the kernel and selected neural grids once using the base seed.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Tuning grids, positive n_configs and the base seed.

    Returns
    -------
    dict of str to list of dict
        Ordered, distinct candidates per bridge, capped by each grid's size.
    """
    if not isinstance(cfg.n_configs, int) or isinstance(cfg.n_configs, bool) or cfg.n_configs < 1:
        raise ValueError("n_configs must be a positive integer.")
    grids = {"kernel": cfg.kernel_grid, "neural": _neural_grid(cfg)}
    result = {}
    for bridge, grid in grids.items():
        if not grid or any(not values for values in grid.values()):
            raise ValueError(f"The {bridge} tuning grid must be nonempty.")
        if any(
            len({fingerprint_settings({"value": v}) for v in values}) != len(values)
            for values in grid.values()
        ):
            raise ValueError(f"The {bridge} tuning grid contains duplicate values.")
        result[bridge] = sample_configs(grid, cfg.n_configs, cfg.seed)
    return result


def tune_resumable(
    folds: Iterable[tuple[sc.AnnData, Evaluate]],
    cfg: DictConfig,
    *,
    directory: Path,
    candidates: dict[str, list[dict]],
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Tune both bridges, committing each completed candidate/fold trial.

    Parameters
    ----------
    folds : iterable of tuple
        Training data and retained-only validation evaluator for each fold.
    cfg : omegaconf.DictConfig
        Effective settings for this seed, including neural_checkpoints.
    directory : pathlib.Path
        Trial storage, owned by a locked run with validated data/config identity.
    candidates : dict of str to list of dict
        Persisted kernel and neural draws; reused across seeds and k values.

    Returns
    -------
    tuple of (pandas.DataFrame, dict, pandas.DataFrame)
        Inner scores, selected settings and mean validation scores of winners.
        Ties use candidate order, then the smallest neural checkpoint.

    Notes
    -----
    Partial neural progress is saved at each scoring checkpoint, but only a
    successfully returned fit is committed. An interrupted trial restarts.
    The caller must validate data identity before reusing this directory.
    """
    checkpoints = list(cfg.neural_checkpoints)
    if (
        not checkpoints
        or any(type(step) is not int or step < 0 for step in checkpoints)
        or len(set(checkpoints)) != len(checkpoints)
    ):
        raise ValueError("neural_checkpoints must be distinct nonnegative integers.")
    checkpoints.sort()
    if set(candidates) != {"kernel", "neural"} or any(not v for v in candidates.values()):
        raise ValueError("Tuning needs nonempty kernel and neural candidates.")
    rows = []
    for fold, (train, evaluate) in enumerate(folds):
        for bridge in ("kernel", "neural"):
            for index, config in enumerate(candidates[bridge]):
                candidate_id = fingerprint_settings(config)
                path = directory / f"fold_{fold}" / bridge / f"{candidate_id}.json"
                settings = (
                    config
                    if bridge == "kernel"
                    else _network_sizes(config) | {"n_steps": max(checkpoints)}
                )
                identity = {
                    "inner_fold": fold,
                    "bridge": bridge,
                    "config": config,
                    "model": asdict(resolve_model_config(cfg, bridge, settings)),
                    "standardise_w": bool(cfg.standardise_w),
                    "checkpoints": checkpoints if bridge == "neural" else [None],
                }
                scores = _resumable_trial(path, identity, train, evaluate, cfg)
                rows.extend(
                    {
                        "inner_fold": fold,
                        "bridge": bridge,
                        **config,
                        "candidate_id": candidate_id,
                        "candidate_index": index,
                        **score,
                    }
                    for score in scores
                )
    if not rows:
        raise ValueError("Tuning requires at least one inner fold.")
    inner = pd.DataFrame(rows).sort_index(axis=1)
    averaged = mean_over_inner_folds(inner).sort_values(
        ["candidate_index", "n_steps"], kind="stable", na_position="first"
    )
    selected = select(averaged, cfg)
    return inner, selected, _winners(averaged)


def _resumable_trial(path, identity, train, evaluate, cfg):
    if path.exists():
        record = read_json(path)
        _validate_trial(record, identity, path)
        print(f"  skip completed trial: {path.relative_to(path.parents[2])}", flush=True)
        return record["rows"]
    rows = []
    partial_path = path.with_suffix(".partial.json")
    bridge, config = identity["bridge"], identity["config"]
    steps = identity["checkpoints"]
    model = (
        make_kernel(config, cfg)
        if bridge == "kernel"
        else make_neural(config | {"n_steps": max(steps)}, cfg)
    )

    def record_score(step):
        score = float(evaluate(model)["mse"])
        if not np.isfinite(score) or score < 0:
            raise ValueError(f"Trial {path} must produce finite, nonnegative MSE.")
        rows.append({"n_steps": step, "mse": score})
        if bridge == "neural":
            write_json(partial_path, {"identity": identity, "rows": rows})
        print(
            f"  {bridge} fold {identity['inner_fold']} checkpoint {step}: mse {score:.6g}",
            flush=True,
        )

    if bridge == "kernel":
        model.fit(train)
        record_score(None)
    else:
        model.fit(train, steps, record_score)
    record = {
        "schema_version": 1,
        "identity": identity,
        "rows": rows,
        "rows_digest": fingerprint_settings({"rows": rows}),
    }
    _validate_trial(record, identity, path)
    write_json(path, record)
    partial_path.unlink(missing_ok=True)
    return rows


def _validate_trial(record, identity, path):
    rows = record.get("rows")
    valid = (
        record.get("schema_version") == 1
        and record.get("identity") == identity
        and isinstance(rows, list)
        and all(isinstance(row, dict) for row in rows)
    )
    if valid:
        valid = (
            [row.get("n_steps") for row in rows] == identity["checkpoints"]
            and all(
                isinstance(row.get("mse"), int | float)
                and np.isfinite(row["mse"])
                and row["mse"] >= 0
                for row in rows
            )
            and record.get("rows_digest") == fingerprint_settings({"rows": rows})
        )
    if not valid:
        raise ValueError(
            f"Invalid trial checkpoint {path}; restore it or use a fresh output directory."
        )
