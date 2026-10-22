"""Out-of-batch prediction of siRNA mean Phenom embeddings on RxRx1, one cell type at a time.

- Data: data/rxrx1_phenom_4x4.h5ad (scripts/embed_rxrx1.py), crops of cfg.cell_type only.
- Batch = experiment; treatment = siRNA; proxy W = the experiment's negative-control crops, plus
  its positive-control crops if cfg.positive_controls_as_proxies.
- Models: the linear ones and the bridges of train_batch_transfer_expression, plus TVN
  (model/tvn.py); a held-out experiment is predicted from its proxy crops alone.
- Folds: K-fold over experiments. Scores per (treatment siRNA, held-out experiment), on means:
  mse      squared error after whitening by the training experiments' within-experiment
           negative-control covariance (utils/metrics.control_whitener): errors in units of typical
           control variation, the same yardstick whichever proxies the models see
  mse_raw  squared error in Phenom coordinates, to check the ranking does not hinge on whitening

Config: configs/train_batch_transfer_rxrx1.yaml. Usage:

    uv run python scripts/train_batch_transfer_rxrx1.py name=huvec 'bridges=[kernel]'
    sbatch -J huvec hpc/run.sbatch scripts/train_batch_transfer_rxrx1.py name=huvec

Prints each fold's mean per (metric, model) as it finishes, siRNAs averaged within each experiment
first. In the Hydra run dir, scores.csv (every score so far) is rewritten after each fold and
summary.csv (the means over all folds) is written at the end.
"""

import time
from collections.abc import Iterator
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from train_batch_transfer_expression import _batch_folds, fit

from genetic_perturbation_playground.data.rxrx1 import load_rxrx1
from genetic_perturbation_playground.model.batch_transfer import (
    _dense,
    batch_labels,
    is_control,
    perturbation_labels,
)
from genetic_perturbation_playground.model.tvn import TVNTransferModel
from genetic_perturbation_playground.utils.metrics import control_whitener, mse


def select_cell_type(adata: sc.AnnData, cell_type: str) -> sc.AnnData:
    """The crops of one cell type."""
    keep = (adata.obs["cell_type"] == cell_type).to_numpy()
    if not keep.any():
        raise ValueError(f"no {cell_type} crops in the RxRx1 embeddings")
    return adata[keep]


def add_positive_controls_to_proxies(adata: sc.AnnData) -> sc.AnnData:
    """A copy of `adata` with the positive-control crops relabelled "control", joining the proxy W."""
    positive = _wells(adata, "positive_control")
    adata = adata.copy()
    adata.obs["perturbation"] = np.where(positive, "control", perturbation_labels(adata))
    return adata


def score_batch(models: dict, test: sc.AnnData, whiten) -> pd.DataFrame:
    """Scores of each model on one held-out experiment: (perturbation, metric, model, value)."""
    labels = perturbation_labels(test)
    present = np.unique(labels[_wells(test, "treatment")])
    perts = np.intersect1d(present, next(iter(models.values())).perturbations_)
    X = _dense(test.X)
    obs = np.stack([X[labels == p].mean(axis=0) for p in perts])
    rows = []
    for name, model in models.items():
        pred = model.predict(perts, test)
        rows.append(_long(perts, name, "mse", mse(whiten(pred), whiten(obs))))
        rows.append(_long(perts, name, "mse_raw", mse(pred, obs)))
    return pd.concat(rows, ignore_index=True)


def _wells(adata: sc.AnnData, well_type: str) -> np.ndarray:
    """Mask of the crops from wells of one type: treatment, negative_control or positive_control."""
    return (adata.obs["well_type"] == well_type).to_numpy()


def _long(perts: np.ndarray, model: str, metric: str, values: np.ndarray) -> pd.DataFrame:
    """One model's per-perturbation values of one metric as long-table rows."""
    return pd.DataFrame({"perturbation": perts, "metric": metric, "model": model, "value": values})


def cross_validate(adata: sc.AnnData, cfg: DictConfig) -> Iterator[pd.DataFrame]:
    """K-fold over experiments, yielding each fold's scores as soon as it is done. Models and the
    metric's whitener are fit on the training experiments."""
    if cfg.positive_controls_as_proxies:
        adata = add_positive_controls_to_proxies(adata)
    print(f"proxy crops: {is_control(adata).sum():,} of {adata.n_obs:,}", flush=True)
    batch = batch_labels(adata)
    negative = _wells(adata, "negative_control")
    folds = _batch_folds(batch, cfg.n_folds, np.random.default_rng(cfg.seed))
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: holding out {len(held_out)} experiments", flush=True)
        is_test = np.isin(batch, held_out)
        train_negative = ~is_test & negative
        whiten = control_whitener(
            _dense(adata.X[train_negative]), batch[train_negative], cfg.whiten_eps
        )
        train = adata[~is_test]
        log_key = f"fold {i + 1}/neural bridge"
        if wandb.run is not None:  # each fold's loss curve against its own step
            wandb.define_metric(f"{log_key}/*", step_metric=f"{log_key}/step")
        fold_cfg = OmegaConf.merge(cfg, {"neural": {"log_key": log_key}})
        start = time.perf_counter()
        models = fit(train, fold_cfg) | {"tvn": TVNTransferModel().fit(train)}
        print(f"  fit {len(models)} models in {time.perf_counter() - start:.0f}s", flush=True)
        scores = [score_batch(models, adata[batch == b], whiten).assign(batch=b) for b in held_out]
        yield pd.concat(scores, ignore_index=True).assign(fold=i)


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean per (model, metric): siRNAs averaged within each experiment, then experiments."""
    per_batch = scores.groupby(["metric", "model", "batch"])["value"].mean()
    return per_batch.groupby(["metric", "model"]).mean().unstack("metric")


@hydra.main(config_path="../configs", config_name="train_batch_transfer_rxrx1", version_base=None)
def main(cfg: DictConfig):
    adata = select_cell_type(load_rxrx1(), cfg.cell_type)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    folds = []
    config = OmegaConf.to_container(cfg)
    with wandb.init(
        project="genetic-playground", name=cfg.name, config=config, mode=cfg.wandb_mode
    ):
        for fold_scores in cross_validate(adata, cfg):
            folds.append(fold_scores)
            print(summary(fold_scores).to_string(float_format="%.4g"), flush=True)
            pd.concat(folds).to_csv(out_dir / "scores.csv", index=False)  # rewritten every fold
    overall = summary(pd.concat(folds))
    overall.to_csv(out_dir / "summary.csv")
    print(f"all folds:\n{overall.to_string(float_format='%.4g')}\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
