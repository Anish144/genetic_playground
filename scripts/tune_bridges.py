"""Nested cross-validation tuning of the kernel and neural bridges on Replogle K562 (or dataset=rxrx1).

Outer folds, control split and scoring are those of train_batch_transfer. Within each outer fold,
a share of the training batches (inner_val_frac) is held out for inner validation. A fixed number
of configurations, sampled at random from each bridge's grid (the same sample in every fold), is
fit on the remaining training batches and scored on the inner-validation batches. The configuration with the lowest mean per-batch MSE is refit on all training batches
and scored on the outer held-out batches, next to the linear models.

Config: configs/tune_bridges.yaml (grids; everything else from train_batch_transfer.yaml). Usage:

    uv run python scripts/tune_bridges.py name=tune

Writes inner_scores.csv (every configuration), selected.csv (winner per outer fold and bridge),
scores_per_perturbation.csv, scores_per_batch.csv and plots/ to the Hydra run dir.
"""

import itertools
from pathlib import Path
from typing import NamedTuple

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from train_batch_transfer import (
    _average_per_batch,
    _average_per_perturbation,
    _batch_folds,
    _fit_linear,
    _plot_scores,
    _print_fold_means,
    _save_scores,
    _score_batches,
    _seen_controls,
    load_data,
    score,
)

from genetic_perturbation_playground.model.batch_transfer import (
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
    batch_labels,
    is_control,
)

INT_SETTINGS = ("n_features", "n_features_w", "n_steps", "cross_fit_folds")


class Split(NamedTuple):
    """Cells to train on and to evaluate on, for one split of the batches."""

    train: sc.AnnData  # training batches, seen controls only
    eval: sc.AnnData  # evaluation batches, all cells
    seen: np.ndarray  # seen-control mask of the evaluation cells
    eval_batches: np.ndarray
    batch: np.ndarray  # batch of every cell in the full data


def _split(adata, batch, seen, train_batches, eval_batches) -> Split:
    """Train on train_batches (seen controls only), evaluate on eval_batches."""
    train_cells = ~is_control(adata) | seen  # models only ever see the seen controls
    is_eval = np.isin(batch, eval_batches)
    train = adata[np.isin(batch, train_batches) & train_cells]
    return Split(train, adata[is_eval], seen[is_eval], eval_batches, batch)


def _grid_configs(grid: DictConfig) -> list[dict]:
    """Every combination of the grid's settings, as {setting: value} dicts."""
    names = list(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())]


def _sample_configs(grid: DictConfig, n: int, seed: int) -> list[dict]:
    """n distinct configurations drawn at random from the grid (all of them if n covers it)."""
    configs = _grid_configs(grid)
    n_drawn = min(n, len(configs))
    picks = np.random.default_rng(seed).choice(len(configs), n_drawn, replace=False)
    return [configs[i] for i in sorted(picks)]


def _make_kernel(config: dict, cfg: DictConfig) -> KernelBridgeTransfer:
    """Kernel bridge with one grid configuration."""
    return KernelBridgeTransfer(config["reg"], cfg.weight_by_cells, config["lengthscale_scale"])


def _make_neural(config: dict, cfg: DictConfig) -> NeuralBridgeTransfer:
    """Neural bridge with one grid configuration (n_steps included)."""
    kwargs = {**cfg.neural, **config}
    return NeuralBridgeTransfer(cfg.weight_by_cells, seed=cfg.seed, **kwargs)


def _validation_scores(model, split: Split, k: int) -> dict:
    """Mean over evaluation batches of each per-batch metric of `model`."""
    scores = _score_batches(
        {"bridge": model}, split.eval, split.seen, split.eval_batches, split.batch, k
    )
    per_batch = _average_per_batch(scores)
    bridge = per_batch[per_batch["model"] == "bridge"]
    return bridge.groupby("metric")["value"].mean().to_dict()


def _tune_kernel(split: Split, cfg: DictConfig) -> list[dict]:
    """Validation scores of every kernel configuration."""
    rows = []
    for config in _sample_configs(cfg.kernel_grid, cfg.n_kernel_configs, cfg.seed):
        model = _make_kernel(config, cfg).fit(split.train)
        rows.append({"bridge": "kernel", **config, **_validation_scores(model, split, cfg.top_k)})
    return rows


def _tune_neural(split: Split, cfg: DictConfig) -> list[dict]:
    """Validation scores of every neural configuration at every checkpoint."""
    checkpoints = list(cfg.neural_checkpoints)
    rows = []
    for config in _sample_configs(cfg.neural_grid, cfg.n_neural_configs, cfg.seed):
        model = _make_neural({**config, "n_steps": max(checkpoints)}, cfg)

        def record(step, model=model, config=config):
            scores = _validation_scores(model, split, cfg.top_k)
            rows.append({"bridge": "neural", **config, "n_steps": step, **scores})
            print(f"  neural {config} step {step}: mse {scores['mse']:.5g}", flush=True)

        model.fit(split.train, checkpoints, record)
    return rows


def _best(inner: pd.DataFrame, bridge: str) -> dict:
    """The configuration of `bridge` with the lowest validation MSE."""
    rows = inner[inner["bridge"] == bridge]
    best = rows.loc[rows["mse"].idxmin()]
    config = best.drop(["bridge", "mse", "pearson", "pearson_top20"]).dropna().to_dict()
    return {key: int(v) if key in INT_SETTINGS else v for key, v in config.items()}


def _fit_tuned(train: sc.AnnData, selected: dict, cfg: DictConfig) -> dict:
    """Linear models plus both bridges with their selected configurations."""
    return _fit_linear(train) | {
        "kernel bridge (tuned)": _make_kernel(selected["kernel"], cfg).fit(train),
        "neural bridge (tuned)": _make_neural(selected["neural"], cfg).fit(train),
    }


def _inner_validation_batches(train_batches: np.ndarray, frac: float, seed) -> np.ndarray:
    """A random share `frac` of the training batches, at least one."""
    n_val = max(1, round(frac * len(train_batches)))
    return np.random.default_rng(seed).permutation(train_batches)[:n_val]


def nested_cv(adata: sc.AnnData, cfg: DictConfig) -> tuple:
    """Tune per outer fold on inner validation, then score the tuned bridges on the outer fold."""
    batch = batch_labels(adata)
    rng = np.random.default_rng(cfg.seed)
    folds = _batch_folds(batch, cfg.n_folds, rng)
    seen = _seen_controls(adata, batch, rng)  # same draws as train_batch_transfer

    inner_rows, selected_rows, pert_scores, batch_scores = [], [], [], []
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: tuning", flush=True)
        train_batches = np.setdiff1d(np.unique(batch), held_out)
        val = _inner_validation_batches(train_batches, cfg.inner_val_frac, [cfg.seed, i])
        inner_split = _split(adata, batch, seen, np.setdiff1d(train_batches, val), val)
        inner = pd.DataFrame(_tune_kernel(inner_split, cfg) + _tune_neural(inner_split, cfg))
        selected = {b: _best(inner, b) for b in ("kernel", "neural")}
        inner_rows.append(inner.assign(fold=i))
        selected_rows += [{"fold": i, "bridge": b, **c} for b, c in selected.items()]
        print(f"selected: {selected}", flush=True)

        outer_split = _split(adata, batch, seen, train_batches, held_out)
        models = _fit_tuned(outer_split.train, selected, cfg)
        pert_scores.append(score(models, outer_split.eval, cfg.top_k, outer_split.seen))
        _print_fold_means(pert_scores[-1])
        batch_scores += _score_batches(
            models, outer_split.eval, outer_split.seen, held_out, batch, cfg.top_k
        )
    return (
        pd.concat(inner_rows),
        pd.DataFrame(selected_rows),
        _average_per_perturbation(pert_scores),
        _average_per_batch(batch_scores),
    )


@hydra.main(config_path="../configs", config_name="tune_bridges", version_base=None)
def main(cfg: DictConfig):
    adata = load_data(cfg)
    inner, selected, per_pert, per_batch = nested_cv(adata, cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    inner.to_csv(out_dir / "inner_scores.csv", index=False)
    selected.to_csv(out_dir / "selected.csv", index=False)
    _save_scores(per_pert, per_batch, out_dir)
    _plot_scores(per_pert, per_batch, out_dir / "plots")


if __name__ == "__main__":
    main()
