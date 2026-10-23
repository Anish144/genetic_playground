"""Nested cross-validation tuning of the kernel and neural bridges on RxRx1, one cell type.

Outer folds, proxies and scores are those of train_batch_transfer_rxrx1 (same seed, same folds).
In each outer fold the training experiments are split into inner_folds folds; each inner fold is
held out for validation once. n_configs configurations per bridge, drawn at random from its grid
(the same draw in every outer fold), are fit on the other inner folds and scored on the held-out
one; each neural configuration is trained once per inner fold to the last of neural_checkpoints
and scored at each. The configuration with the lowest validation mse averaged over the inner
folds is refit on all training experiments and scored on the outer fold, next to the linear
models and TVN. shift (b=1) is scored on the inner folds too, as a reference only.
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
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from train_batch_transfer_expression import _fit_linear
from train_batch_transfer_rxrx1 import (
    add_positive_controls_to_proxies,
    experiment_folds,
    log_summary,
    score_batches,
    select_cell_type,
    summary,
)

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.data.rxrx1 import load_rxrx1
from genetic_perturbation_playground.model.batch_transfer import batch_labels
from genetic_perturbation_playground.model.tvn import TVNTransferModel


def validation_scores(model, val: sc.AnnData) -> dict:
    """mse of `model` on `val`: siRNAs averaged within each experiment, then experiments."""
    scores = score_batches({"bridge": model}, val, np.unique(batch_labels(val)))
    return summary(scores).loc["bridge"].to_dict()


def inner_folds(adata: sc.AnnData, train_batches: np.ndarray, cfg: DictConfig, seed) -> Iterator:
    """(training data, Evaluate) of each of cfg.inner_folds folds of `train_batches`, fit on the
    other folds."""
    batch = batch_labels(adata)
    rng = np.random.default_rng(seed)
    for val_batches in experiment_folds(train_batches, cfg.inner_folds, rng, cfg.cell_type):
        train = adata[np.isin(batch, np.setdiff1d(train_batches, val_batches))]
        yield train, partial(validation_scores, val=adata[np.isin(batch, val_batches)])


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
    folds = experiment_folds(batch, cfg.n_folds, np.random.default_rng(cfg.seed), cfg.cell_type)
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: tuning", flush=True)
        train_batches = np.setdiff1d(np.unique(batch), held_out)
        inner, selected, winners = tuning.tune(
            inner_folds(adata, train_batches, cfg, [cfg.seed, i]), cfg
        )
        log_selected(i + 1, selected)
        log_summary(f"fold {i + 1}/validation", winners)
        train = adata[np.isin(batch, train_batches)]
        models = _fit_linear(train) | {"tvn": TVNTransferModel().fit(train)}
        models |= tuning.fit_selected(train, selected, cfg, f"fold {i + 1}/neural bridge")
        yield (
            inner.assign(fold=i),
            pd.DataFrame([{"fold": i, "bridge": b, **s} for b, s in selected.items()]),
            score_batches(models, adata, held_out).assign(fold=i),
        )


@hydra.main(config_path="../configs", config_name="tune_bridges_rxrx1", version_base=None)
def main(cfg: DictConfig):
    adata = select_cell_type(load_rxrx1(**cfg.embeddings), cfg.cell_type)
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
