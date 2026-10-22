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
from train_batch_transfer_expression import _batch_folds, _fit_linear, _make_bridge, _print_slope

from genetic_perturbation_playground.data.rxrx1 import load_rxrx1
from genetic_perturbation_playground.model.batch_transfer import (
    _dense,
    batch_labels,
    is_control,
    perturbation_labels,
)
from genetic_perturbation_playground.model.tvn import TVNTransferModel
from genetic_perturbation_playground.utils.metrics import (
    centre_within_batches,
    control_whitener,
    mse,
)


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


class Standardised:
    """`model` fit and predicting on embeddings standardised per dimension; predictions mapped back.

    Standardises W (control crops) and Y (siRNA means) alike, so no few high-variance dimensions
    dominate a bridge's squared-error loss. Mean: the training negative controls'; std: their
    typical within-experiment variation (each experiment centred on its own mean), the diagonal of
    the metric's whitener.
    """

    def __init__(self, model):
        self.model = model

    @property
    def perturbations_(self) -> np.ndarray:
        return self.model.perturbations_

    def fit(self, adata: sc.AnnData, *args) -> "Standardised":
        """Fit `model` on standardised `adata`; extra arguments go to its fit (checkpoints)."""
        negative = _wells(adata, "negative_control")
        controls = _dense(adata.X[negative])
        self.mean_ = controls.mean(axis=0)
        self.std_ = centre_within_batches(controls, batch_labels(adata)[negative]).std(axis=0)
        self.model.fit(self._standardise(adata), *args)
        return self

    def predict(self, perturbations: list[str], test: sc.AnnData) -> np.ndarray:
        """`model`'s prediction from standardised `test`, mapped back to embedding coordinates."""
        return self.model.predict(perturbations, self._standardise(test)) * self.std_ + self.mean_

    def _standardise(self, adata: sc.AnnData) -> sc.AnnData:
        return sc.AnnData((_dense(adata.X) - self.mean_) / self.std_, obs=adata.obs)


def with_standardisation(bridge, cfg: DictConfig):
    """`bridge` (unfitted), wrapped in Standardised if cfg.standardise_bridges."""
    return Standardised(bridge) if cfg.standardise_bridges else bridge


def fit_models(train: sc.AnnData, cfg: DictConfig) -> dict:
    """Linear models, TVN and the bridges of cfg.bridges, fit on `train`."""
    models = _fit_linear(train) | {"tvn": TVNTransferModel().fit(train)}
    _print_slope(models["transfer (fit b)"].slope_)
    bridges = {f"{b} bridge": with_standardisation(_make_bridge(b, cfg), cfg) for b in cfg.bridges}
    return models | {name: bridge.fit(train) for name, bridge in bridges.items()}


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


def score_batches(models: dict, adata: sc.AnnData, batches, whiten) -> pd.DataFrame:
    """score_batch on each of `batches` of `adata`, tagged with its batch."""
    batch = batch_labels(adata)
    return pd.concat(
        [score_batch(models, adata[batch == b], whiten).assign(batch=b) for b in batches],
        ignore_index=True,
    )


def _wells(adata: sc.AnnData, well_type: str) -> np.ndarray:
    """Mask of the crops from wells of one type: treatment, negative_control or positive_control."""
    return (adata.obs["well_type"] == well_type).to_numpy()


def negative_control_whitener(train: sc.AnnData, eps: float):
    """The metric's whitener (utils/metrics.control_whitener) from the negative controls of `train`."""
    negative = _wells(train, "negative_control")
    return control_whitener(_dense(train.X[negative]), batch_labels(train)[negative], eps)


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
    folds = _batch_folds(batch, cfg.n_folds, np.random.default_rng(cfg.seed))
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: holding out {len(held_out)} experiments", flush=True)
        train = adata[~np.isin(batch, held_out)]
        whiten = negative_control_whitener(train, cfg.whiten_eps)
        fold_cfg = OmegaConf.merge(cfg, {"neural": {"log_key": f"fold {i + 1}/neural bridge"}})
        start = time.perf_counter()
        models = fit_models(train, fold_cfg)
        print(f"  fit {len(models)} models in {time.perf_counter() - start:.0f}s", flush=True)
        yield score_batches(models, adata, held_out, whiten).assign(fold=i)


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean per (model, metric): siRNAs averaged within each experiment, then experiments."""
    per_batch = scores.groupby(["metric", "model", "batch"])["value"].mean()
    return per_batch.groupby(["metric", "model"]).mean().unstack("metric")


def log_summary(prefix: str, table: pd.DataFrame) -> None:
    """Every value of a summary() table into the open W&B run's summary, as <prefix>/<model>/<metric>."""
    if wandb.run is not None:
        wandb.summary.update(
            {f"{prefix}/{m}/{metric}": v for (m, metric), v in table.stack().items()}
        )


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
            fold_summary = summary(fold_scores)
            print(fold_summary.to_string(float_format="%.4g"), flush=True)
            log_summary(f"fold {len(folds)}", fold_summary)
            pd.concat(folds).to_csv(out_dir / "scores.csv", index=False)  # rewritten every fold
        overall = summary(pd.concat(folds))
        log_summary("all folds", overall)
    overall.to_csv(out_dir / "summary.csv")
    print(f"all folds:\n{overall.to_string(float_format='%.4g')}\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
