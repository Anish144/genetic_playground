"""Nested cross-validation tuning of the kernel and neural bridges on Replogle K562.

Select n_validation_batches distinct outer batches using seed, as in train_batch_transfer_expression.
Each is held out separately; all remaining batches are split into inner_folds folds for tuning.
The configuration with the lowest mean inner-validation mse is refit on all remaining batches
and scored on the one outer batch, next to the linear models. shift (b=1) is scored on the inner
folds too, as a reference only. The outer batch never enters tuning. n_validation_batches counts
outer evaluations; the inner tuning fits and final refits are additional training runs.

Config: configs/tune_bridges.yaml (grids; everything else from train_batch_transfer_expression.yaml).
Usage:

    sbatch -J tune --time=2-00:00:00 hpc/run.sbatch scripts/tune_bridges.py name=tune
    sbatch -J perx --time=2-00:00:00 hpc/run.sbatch scripts/tune_bridges.py \
        --config-name tune_bridges_perx name=perx

Writes inner_scores.csv (every configuration on every inner fold), selected.csv (winner per outer
fold and bridge), scores_per_perturbation.csv, scores_per_batch.csv, scores_per_pair.csv (each
model's mse, the squared error averaged over genes, per (perturbation, held-out batch)) and plots/
to the Hydra run dir. Per-perturbation and per-pair scores include n_training_batches from the
outer refit. plots/by_training_batches/batch_<label>/ contains a separate scatter plot per metric
and per_perturbation.csv for each held-out batch, using that batch's scores directly.
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
from genetic_perturbation_playground.model.batch_transfer import (
    batch_labels,
    is_control,
    training_batch_counts,
)


class Split(NamedTuple):
    """Cells to train on and to evaluate on, for one split of the batches."""

    train: sc.AnnData  # training batches, seen controls only
    eval: sc.AnnData  # evaluation batches, all cells
    seen: np.ndarray  # seen-control mask of the evaluation cells


def _split(adata, batch, seen, train_batches, eval_batches) -> Split:
    """Train on train_batches (seen controls only), evaluate on eval_batches."""
    train_cells = ~is_control(adata) | seen  # models only ever see the seen controls
    is_eval = np.isin(batch, eval_batches)
    train = adata[np.isin(batch, train_batches) & train_cells]
    return Split(train, adata[is_eval], seen[is_eval])


def _validation_scores(model, split: Split, k: int) -> dict:
    """Mean over evaluation batches of each per-batch metric of `model`."""
    scores = _score_batches({"bridge": model}, split.eval, split.seen, k)
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
    """Tune and evaluate on n_validation_batches distinct leave-one-batch-out splits.

    For each selected outer batch, tune on inner folds of all other batches, refit on those
    batches, and score only the outer batch. Inner tuning retains cfg.inner_folds K-fold splits.

    Returns (inner scores, selected settings, per-perturbation averages, per-batch averages,
    individual scores for every metric per (perturbation, held-out batch, model)).
    """
    batch = batch_labels(adata)
    rng = np.random.default_rng(cfg.seed)
    selected_batches = tuning.validation_batches(batch, cfg.n_validation_batches, rng)
    seen = _seen_controls(adata, batch, rng)  # same draws as train_batch_transfer_expression

    inner_tables, selected_rows, pert_scores, batch_scores = [], [], [], []
    for i, held_out in enumerate(selected_batches):
        print(
            f"validation {i + 1}/{len(selected_batches)}: tuning with {held_out} held out",
            flush=True,
        )
        train_batches = np.setdiff1d(np.unique(batch), held_out)
        folds_i = _inner_folds(adata, batch, seen, train_batches, cfg, [cfg.seed, i])
        inner, selected, _ = tuning.tune(folds_i, cfg)
        inner_tables.append(inner.assign(fold=i))
        selected_rows += [{"fold": i, "bridge": b, **c} for b, c in selected.items()]

        outer = _split(adata, batch, seen, train_batches, [held_out])
        models = _fit_linear(outer.train, cfg.standardise_w) | tuning.fit_selected(
            outer.train, selected, cfg
        )
        scores = score(models, outer.eval, cfg.top_k, outer.seen)
        scores["n_training_batches"] = scores["perturbation"].map(
            training_batch_counts(outer.train)
        )
        pert_scores.append(scores)
        _print_fold_means(pert_scores[-1])
        batch_scores.append(scores.assign(batch=held_out, fold=i, cells=outer.eval.n_obs))
    pair_scores = pd.concat(batch_scores, ignore_index=True)
    return (
        pd.concat(inner_tables),
        pd.DataFrame(selected_rows),
        _average_per_perturbation(pert_scores),
        _average_per_batch(batch_scores),
        pair_scores.drop(columns="cells"),  # cells is the whole batch's, not the pair's
    )


@hydra.main(config_path="../configs", config_name="tune_bridges", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    inner, selected, per_pert, per_batch, per_pair = nested_cv(adata, cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    inner.to_csv(out_dir / "inner_scores.csv", index=False)
    selected.to_csv(out_dir / "selected.csv", index=False)
    _save_scores(per_pert, per_batch, out_dir)
    per_pair.query("metric == 'mse'").drop(columns="metric").to_csv(
        out_dir / "scores_per_pair.csv", index=False
    )
    _plot_scores(per_pert, per_batch, per_pair, out_dir / "plots")


if __name__ == "__main__":
    main()
