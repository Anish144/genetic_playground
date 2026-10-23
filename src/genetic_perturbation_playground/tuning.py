"""Folds over batches and random-search tuning of the kernel and neural bridges.

Shared by scripts/tune_bridges.py (Replogle) and scripts/tune_bridges_rxrx1.py. cfg.n_configs
configurations are drawn from each bridge's grid, the same draw in every fold. On each inner fold,
shift (b=1, a reference only), every kernel configuration and every neural configuration at each
of cfg.neural_checkpoints are fit on the other inner folds and scored on this one. The
configuration with the lowest validation mse averaged over the inner folds is selected.
"""

import itertools
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig

from genetic_perturbation_playground.model.batch_transfer import (
    BatchTransferModel,
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
    Standardised,
)

METRICS = ["mse", "pearson", "pearson_top20"]  # of either data set
INT_SETTINGS = ["phi_size", "psi_size", "n_steps", "depth", "cross_fit_folds"]
Evaluate = Callable[[object], dict]  # fitted model -> its validation scores, {metric: value}


def batch_folds(batch: np.ndarray, n_folds: int, rng: np.random.Generator) -> list:
    """The distinct batches of `batch`, shuffled and split into n_folds groups."""
    return np.array_split(rng.permutation(np.unique(batch)), n_folds)


def sample_configs(grid: DictConfig, n: int, seed: int) -> list[dict]:
    """n distinct configurations drawn at random from the grid (all of them if n covers it)."""
    configs = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]
    picks = np.random.default_rng(seed).choice(len(configs), min(n, len(configs)), replace=False)
    return [configs[i] for i in sorted(picks)]


def with_standardisation(bridge, cfg: DictConfig):
    """`bridge` (unfitted), wrapped in Standardised if cfg.standardise_bridges."""
    return Standardised(bridge) if cfg.standardise_bridges else bridge


def make_kernel(config: dict, cfg: DictConfig):
    """Unfitted kernel bridge with one configuration: reg and lengthscale_scale."""
    kernel = KernelBridgeTransfer(config["reg"], cfg.weight_by_cells, config["lengthscale_scale"])
    return with_standardisation(kernel, cfg)


def make_neural(config: dict, cfg: DictConfig):
    """Unfitted neural bridge with one configuration; the rest of its settings from cfg.neural."""
    settings = {**cfg.neural, **_network_sizes(config)}
    neural = NeuralBridgeTransfer(cfg.weight_by_cells, seed=cfg.seed, **settings)
    return with_standardisation(neural, cfg)


def _network_sizes(config: dict) -> dict:
    """`config` with phi_size as phi's hidden width and output dimension, psi_size as psi's."""
    settings = dict(config)
    phi, psi = settings.pop("phi_size"), settings.pop("psi_size")
    return settings | {"width_x": phi, "n_features": phi, "width_w": psi, "n_features_w": psi}


def tune(folds: Iterable[tuple[sc.AnnData, Evaluate]], cfg: DictConfig) -> tuple:
    """Tune both bridges on the inner folds, each given as (training data, its Evaluate).

    Returns the validation scores on every inner fold, the selected configuration of each bridge,
    and the mean validation scores of the selected configurations and of shift.
    """
    inner = _inner_scores(folds, cfg)
    averaged = mean_over_inner_folds(inner)
    selected, winners = select(averaged), _winners(averaged)
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
        _shift_rows(train, evaluate)
        + _kernel_rows(train, evaluate, cfg)
        + _neural_rows(train, evaluate, cfg)
    )


def _shift_rows(train: sc.AnnData, evaluate: Evaluate) -> list[dict]:
    """Validation scores of shift (b=1), the reference the bridges are compared with."""
    scores = evaluate(BatchTransferModel(1.0).fit(train))
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
    for config in sample_configs(cfg.neural_grid, cfg.n_configs, cfg.seed):
        model = make_neural(config | {"n_steps": max(checkpoints)}, cfg)

        def record(step, model=model, config=config):
            scores = evaluate(model)
            rows.append({"bridge": "neural", **config, "n_steps": step, **scores})
            print(f"  neural {config} step {step}: mse {scores['mse']:.4g}", flush=True)

        model.fit(train, checkpoints, record)
    return rows


def mean_over_inner_folds(inner: pd.DataFrame) -> pd.DataFrame:
    """Each configuration's (and shift's) validation scores averaged over the inner folds."""
    metrics = [c for c in inner.columns if c in METRICS]
    settings = [c for c in inner.columns if c not in [*metrics, "inner_fold"]]
    return inner.groupby(settings, dropna=False)[metrics].mean().reset_index()


def select(averaged: pd.DataFrame) -> dict:
    """The configuration of each bridge with the lowest mean validation mse."""
    return {
        bridge: _best(averaged[averaged["bridge"] == bridge]) for bridge in ["kernel", "neural"]
    }


def _winners(averaged: pd.DataFrame) -> pd.DataFrame:
    """Mean validation scores of the best configuration of each bridge, and of shift."""
    best = averaged.loc[averaged.groupby("bridge")["mse"].idxmin()].set_index("bridge")
    return best[[m for m in METRICS if m in best]]


def _best(rows: pd.DataFrame) -> dict:
    """The settings of the row with the lowest mse, integer settings as ints."""
    best = rows.loc[rows["mse"].idxmin()].drop(["bridge", *METRICS], errors="ignore").dropna()
    return {key: int(v) if key in INT_SETTINGS else v for key, v in best.items()}


def fit_selected(train: sc.AnnData, selected: dict, cfg: DictConfig, log_key=None) -> dict:
    """Both bridges in their selected configurations, fit on `train`; the neural bridge logs its
    loss to W&B under `log_key` if one is given."""
    kernel = make_kernel(selected["kernel"], cfg)
    neural = make_neural(selected["neural"] | {"log_key": log_key}, cfg)
    return {"kernel bridge (tuned)": kernel.fit(train), "neural bridge (tuned)": neural.fit(train)}
