"""Nested cross-validation tuning of the kernel and neural bridges on RxRx1, one cell type.

Outer folds, proxies and scores are those of train_batch_transfer_rxrx1 (same seed, same folds).
In each outer fold the training experiments are split into inner_folds folds; each inner fold is
held out for validation once. n_configs configurations per bridge, drawn at random from its grid
(the same draw in every outer fold), are fit on the other inner folds and scored on the held-out
one; each neural configuration is trained once per inner fold to the last of neural_checkpoints
and scored at each. The configuration with the lowest validation mse averaged over the inner
folds is refit on all training experiments and scored on the outer fold, next to the linear
models and TVN. shift (b=1) is scored on the inner folds too, as a reference only. Each score's
whitener comes from the experiments its models were fit on.

W&B: one run per job, named name. Per outer fold k: the selected settings (fold k/selected/...),
their mean validation scores and shift's (fold k/validation/<bridge>/<metric>), the training
loss of the refit neural bridge (fold k/neural bridge/loss, against its step) and every model's
outer scores (fold k/<model>/<metric>); at the end, the means over all folds
(all folds/<model>/<metric>). The tuning fits are not logged; inner_scores.csv holds them.

Config: configs/tune_bridges_rxrx1.yaml. Usage:

    sbatch -J tune_huvec --time=24:00:00 hpc/run.sbatch scripts/tune_bridges_rxrx1.py \
        name=tune_huvec positive_controls_as_proxies=true

Writes inner_scores.csv (every configuration on every inner fold), selected.csv (winner per fold
and bridge) and scores.csv (outer scores) after every fold, and summary.csv at the end, to the
Hydra run dir.
"""

from collections.abc import Iterator
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from train_batch_transfer_expression import _batch_folds, _fit_linear
from train_batch_transfer_rxrx1 import (
    add_positive_controls_to_proxies,
    log_summary,
    negative_control_whitener,
    score_batches,
    select_cell_type,
    summary,
    with_standardisation,
)
from tune_bridges import (
    _best,
    _make_kernel,
    _make_neural,
    _sample_configs,
)

from genetic_perturbation_playground.data.rxrx1 import load_rxrx1
from genetic_perturbation_playground.model.batch_transfer import BatchTransferModel, batch_labels
from genetic_perturbation_playground.model.tvn import TVNTransferModel


def validation_scores(model, val: sc.AnnData, whiten) -> dict:
    """mse and mse_raw of `model` on `val`: siRNAs averaged within each experiment, then experiments."""
    scores = score_batches({"bridge": model}, val, np.unique(batch_labels(val)), whiten)
    return summary(scores).loc["bridge"].to_dict()


def shift_reference(train: sc.AnnData, val: sc.AnnData, whiten) -> list[dict]:
    """Validation scores of shift (b=1), the reference the bridges are compared with."""
    scores = validation_scores(BatchTransferModel(1.0).fit(train), val, whiten)
    print(f"  shift (b=1): mse {scores['mse']:.4g}", flush=True)
    return [{"bridge": "shift (b=1)", **scores}]


def tune_kernel(train: sc.AnnData, val: sc.AnnData, whiten, cfg: DictConfig) -> list[dict]:
    """Validation scores of each sampled kernel configuration."""
    rows = []
    for config in _sample_configs(cfg.kernel_grid, cfg.n_configs, cfg.seed):
        kernel = with_standardisation(_make_kernel(config, cfg), cfg)
        scores = validation_scores(kernel.fit(train), val, whiten)
        rows.append({"bridge": "kernel", **config, **scores})
        print(f"  kernel {config}: mse {scores['mse']:.4g}", flush=True)
    return rows


def tune_neural(train: sc.AnnData, val: sc.AnnData, whiten, cfg: DictConfig) -> list[dict]:
    """Validation scores of each sampled neural configuration at every checkpoint."""
    checkpoints = list(cfg.neural_checkpoints)
    rows = []
    for config in _sample_configs(cfg.neural_grid, cfg.n_configs, cfg.seed):
        neural = _make_neural(network_sizes(config) | {"n_steps": max(checkpoints)}, cfg)
        model = with_standardisation(neural, cfg)

        def record(step, model=model, config=config):
            scores = validation_scores(model, val, whiten)
            rows.append({"bridge": "neural", **config, "n_steps": step, **scores})
            print(f"  neural {config} step {step}: mse {scores['mse']:.4g}", flush=True)

        model.fit(train, checkpoints, record)
    return rows


def inner_scores(
    adata: sc.AnnData, train_batches: np.ndarray, cfg: DictConfig, seed
) -> pd.DataFrame:
    """Validation scores of shift and every bridge configuration on each of cfg.inner_folds folds
    of `train_batches`, fit on the others."""
    batch = batch_labels(adata)
    folds = _batch_folds(train_batches, cfg.inner_folds, np.random.default_rng(seed))
    tables = []
    for j, val_batches in enumerate(folds):
        print(f"  inner fold {j + 1}/{cfg.inner_folds}", flush=True)
        train = adata[np.isin(batch, np.setdiff1d(train_batches, val_batches))]
        val = adata[np.isin(batch, val_batches)]
        whiten = negative_control_whitener(train, cfg.whiten_eps)
        rows = shift_reference(train, val, whiten)
        rows += tune_kernel(train, val, whiten, cfg) + tune_neural(train, val, whiten, cfg)
        tables.append(pd.DataFrame(rows).assign(inner_fold=j))
    return pd.concat(tables, ignore_index=True)


def mean_over_inner_folds(inner: pd.DataFrame) -> pd.DataFrame:
    """Each configuration's (and shift's) validation scores averaged over the inner folds."""
    metrics = ["mse", "mse_raw"]
    settings = [c for c in inner.columns if c not in [*metrics, "inner_fold"]]
    return inner.groupby(settings, dropna=False)[metrics].mean().reset_index()


def network_sizes(config: dict) -> dict:
    """Neural bridge settings of a grid configuration: phi_size sets phi's hidden width and output
    dimension, psi_size those of psi."""
    settings = dict(config)
    phi, psi = settings.pop("phi_size"), settings.pop("psi_size")
    return settings | {"width_x": phi, "n_features": phi, "width_w": psi, "n_features_w": psi}


def fit_tuned(train: sc.AnnData, selected: dict, cfg: DictConfig, log_key: str) -> dict:
    """Linear models, TVN and both bridges in their selected configurations, fit on `train`; the
    neural bridge logs its loss under `log_key`."""
    neural = _make_neural(network_sizes(selected["neural"]) | {"log_key": log_key}, cfg)
    kernel = _make_kernel(selected["kernel"], cfg)
    return _fit_linear(train) | {
        "tvn": TVNTransferModel().fit(train),
        "kernel bridge (tuned)": with_standardisation(kernel, cfg).fit(train),
        "neural bridge (tuned)": with_standardisation(neural, cfg).fit(train),
    }


def log_selected(fold: int, selected: dict) -> None:
    """One fold's selected settings into the open W&B run's summary, as fold <k>/selected/..."""
    if wandb.run is not None:
        wandb.summary.update(
            {f"fold {fold}/selected/{b}/{k}": v for b, s in selected.items() for k, v in s.items()}
        )


def nested_cv(adata: sc.AnnData, cfg: DictConfig) -> Iterator[tuple]:
    """Per outer fold: (inner scores, selected settings, outer scores), as soon as it is done."""
    if cfg.positive_controls_as_proxies:
        adata = add_positive_controls_to_proxies(adata)
    batch = batch_labels(adata)
    folds = _batch_folds(batch, cfg.n_folds, np.random.default_rng(cfg.seed))
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: tuning", flush=True)
        train_batches = np.setdiff1d(np.unique(batch), held_out)
        inner = inner_scores(adata, train_batches, cfg, [cfg.seed, i])
        averaged = mean_over_inner_folds(inner)
        selected = {bridge: _best(averaged, bridge) for bridge in ["kernel", "neural"]}
        winners = averaged.loc[averaged.groupby("bridge")["mse"].idxmin()]
        winners = winners.set_index("bridge")[["mse", "mse_raw"]]
        print(f"fold {i + 1}/{cfg.n_folds}: selected {selected}", flush=True)
        print(winners.to_string(float_format="%.4g"), flush=True)
        log_selected(i + 1, selected)
        log_summary(f"fold {i + 1}/validation", winners)
        train = adata[np.isin(batch, train_batches)]
        models = fit_tuned(train, selected, cfg, f"fold {i + 1}/neural bridge")
        whiten = negative_control_whitener(train, cfg.whiten_eps)
        scores = score_batches(models, adata, held_out, whiten)
        yield (
            inner.assign(fold=i),
            pd.DataFrame([{"fold": i, "bridge": b, **s} for b, s in selected.items()]),
            scores.assign(fold=i),
        )


@hydra.main(config_path="../configs", config_name="tune_bridges_rxrx1", version_base=None)
def main(cfg: DictConfig):
    adata = select_cell_type(load_rxrx1(), cfg.cell_type)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    tables = {"inner_scores": [], "selected": [], "scores": []}
    config = OmegaConf.to_container(cfg)
    with wandb.init(
        project="genetic-playground", name=cfg.name, config=config, mode=cfg.wandb_mode
    ):
        for fold, fold_tables in enumerate(nested_cv(adata, cfg), start=1):
            for rows, table in zip(tables.values(), fold_tables, strict=True):
                rows.append(table)
            for name, rows in tables.items():  # rewritten every fold
                pd.concat(rows).to_csv(out_dir / f"{name}.csv", index=False)
            fold_summary = summary(fold_tables[-1])
            print(fold_summary.to_string(float_format="%.4g"), flush=True)
            log_summary(f"fold {fold}", fold_summary)
        overall = summary(pd.concat(tables["scores"]))
        log_summary("all folds", overall)
    overall.to_csv(out_dir / "summary.csv")
    print(f"all folds:\n{overall.to_string(float_format='%.4g')}\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
