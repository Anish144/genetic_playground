"""Nested cross-validation tuning of the kernel and neural bridges on Replogle K562.

Outer folds, control split and scoring are those of train_batch_transfer_expression (same seed,
same folds). In each outer fold the training batches are split into inner_folds folds and the
bridges are tuned on them as described in genetic_perturbation_playground/tuning.py: the
configuration with the lowest validation mse averaged over the inner folds is refit on all
training batches and scored on the outer fold, next to the linear models. shift (b=1) is scored
on the inner folds too, as a reference only.

Config: configs/tune_bridges.yaml (grids; everything else from train_batch_transfer_expression.yaml).
Usage:

    sbatch -J tune --time=6-00:00:00 hpc/run.sbatch scripts/tune_bridges.py name=tune

Writes inner_scores.csv (every configuration on every inner fold), selected.csv (winner per outer
fold and bridge), scores_per_perturbation.csv, scores_per_batch.csv and plots/ to the Hydra run dir.
"""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import NamedTuple

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from train_batch_transfer_expression import (
    _average_per_batch,
    _average_per_perturbation,
    _fit_linear,
    _plot_scores,
    _print_fold_means,
    _save_scores,
    _score_batches,
    _seen_controls,
    score,
)

from genetic_perturbation_playground import tuning
from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.batch_transfer import batch_labels, is_control


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


def _validation_scores(model, split: Split, k: int) -> dict:
    """Mean over evaluation batches of each per-batch metric of `model`."""
    scores = _score_batches(
        {"bridge": model}, split.eval, split.seen, split.eval_batches, split.batch, k
    )
    per_batch = _average_per_batch(scores)
    bridge = per_batch[per_batch["model"] == "bridge"]
    return bridge.groupby("metric")["value"].mean().to_dict()


def _inner_folds(adata, batch, seen, train_batches, cfg: DictConfig, seed) -> Iterator:
    """(training data, Evaluate) of each of cfg.inner_folds folds of `train_batches`, fit on the
    other folds."""
    rng = np.random.default_rng(seed)
    for val_batches in tuning.batch_folds(train_batches, cfg.inner_folds, rng):
        split = _split(adata, batch, seen, np.setdiff1d(train_batches, val_batches), val_batches)
        yield split.train, partial(_validation_scores, split=split, k=cfg.top_k)


def nested_cv(adata: sc.AnnData, cfg: DictConfig) -> tuple:
    """Tune per outer fold on its inner folds, then score the tuned bridges on the outer fold.

    Returns (inner scores, selected settings, per-perturbation scores, per-batch scores).
    """
    batch = batch_labels(adata)
    rng = np.random.default_rng(cfg.seed)
    folds = tuning.batch_folds(batch, cfg.n_folds, rng)
    seen = _seen_controls(adata, batch, rng)  # same draws as train_batch_transfer_expression

    inner_tables, selected_rows, pert_scores, batch_scores = [], [], [], []
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: tuning", flush=True)
        train_batches = np.setdiff1d(np.unique(batch), held_out)
        folds_i = _inner_folds(adata, batch, seen, train_batches, cfg, [cfg.seed, i])
        inner, selected, _ = tuning.tune(folds_i, cfg)
        inner_tables.append(inner.assign(fold=i))
        selected_rows += [{"fold": i, "bridge": b, **c} for b, c in selected.items()]

        outer = _split(adata, batch, seen, train_batches, held_out)
        models = _fit_linear(outer.train) | tuning.fit_selected(outer.train, selected, cfg)
        pert_scores.append(score(models, outer.eval, cfg.top_k, outer.seen))
        _print_fold_means(pert_scores[-1])
        batch_scores += _score_batches(models, outer.eval, outer.seen, held_out, batch, cfg.top_k)
    return (
        pd.concat(inner_tables),
        pd.DataFrame(selected_rows),
        _average_per_perturbation(pert_scores),
        _average_per_batch(batch_scores),
    )


@hydra.main(config_path="../configs", config_name="tune_bridges", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    inner, selected, per_pert, per_batch = nested_cv(adata, cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    inner.to_csv(out_dir / "inner_scores.csv", index=False)
    selected.to_csv(out_dir / "selected.csv", index=False)
    _save_scores(per_pert, per_batch, out_dir)
    _plot_scores(per_pert, per_batch, out_dir / "plots")


if __name__ == "__main__":
    main()
