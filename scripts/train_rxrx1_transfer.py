"""Train and evaluate transfer models on held-out RxRx1 batches.

- Data: data/rxrx1_phenom_4x4.h5ad (scripts/embed_rxrx1.py), crops of cfg.cell_type only.
- Batch = plate within experiment (default), or experiment, selected by cfg.batch_unit.
- Treatment = siRNA; proxy W = the batch's positive-control crops (default), negative-control
  crops, or both, selected by cfg.proxy_controls.
- Models: the linear ones and the bridges of train_replogle_transfer, with scaled inputs
  and unscaled targets; a held-out batch is predicted from its proxy crops alone.
- Validation: select n_validation_batches distinct batches without replacement using seed.
  Hold out each one separately and fit each model on every other batch, for
  n_validation_batches fits per model. Scores per (treatment siRNA, held-out batch), on means:
  mse      squared error in Phenom coordinates

Config: configs/train_rxrx1_transfer.yaml. Usage:

    uv run python scripts/train_rxrx1_transfer.py name=huvec 'bridges=[kernel]'
    uv run python scripts/train_rxrx1_transfer.py name=hybrid neural=hybrid
    uv run python scripts/train_rxrx1_transfer.py name=experiments batch_unit=experiment
    sbatch -J huvec hpc/run.sbatch scripts/train_rxrx1_transfer.py name=huvec

Prints each validation run's mean per (metric, model) as it finishes. In the Hydra run dir,
scores.csv is rewritten after each run and summary.csv averages over the selected batches.
The existing "fold" column and W&B keys identify the leave-one-batch-out runs.
Scores include n_training_batches, counting training batches containing each siRNA.
plots/by_training_batches/batch_<label>/ contains a separate scatter plot per metric and
per_perturbation.csv for each held-out batch, using that batch's scores directly.
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
from train_replogle_transfer import fit as fit_models

from genetic_perturbation_playground.config import register_configs
from genetic_perturbation_playground.data.rxrx1 import load_rxrx1
from genetic_perturbation_playground.data.transfer import (
    as_dense,
    batch_labels,
    is_control,
    perturbation_labels,
    training_batch_counts,
)
from genetic_perturbation_playground.tuning import batch_folds, validation_batches
from genetic_perturbation_playground.utils.metrics import mse
from genetic_perturbation_playground.utils.plotting import report_by_training_batches

register_configs()


def select_cell_type(adata: sc.AnnData, cell_type: str | None) -> sc.AnnData:
    """The crops of one cell type; with None, of all of them, each (cell type, siRNA) a perturbation
    of its own ("<cell type>|<siRNA>"), since each experiment holds one cell type."""
    if cell_type is None:
        adata, labels = adata.copy(), perturbation_labels(adata)
        cell_types = adata.obs["cell_type"].astype(str).to_numpy()
        adata.obs["perturbation"] = np.where(
            labels == "control", "control", np.char.add(cell_types + "|", labels)
        )
        return adata
    keep = (adata.obs["cell_type"] == cell_type).to_numpy()
    if not keep.any():
        raise ValueError(f"no {cell_type} crops in the RxRx1 embeddings")
    return adata[keep]


def select_batch_unit(adata: sc.AnnData, unit: str) -> sc.AnnData:
    """Use experiments or (experiment, plate) pairs as batches without changing the input.

    Embeddings store the experiment in batch. Preserve it as experiment_batch when relabelling
    so repeated calls do not append plate suffixes and experiment batches can be restored.
    """
    if unit not in ("experiment", "plate"):
        raise ValueError("batch_unit must be 'experiment' or 'plate'")
    if unit == "experiment" and "experiment_batch" not in adata.obs:
        return adata
    if unit == "plate" and ("plate" not in adata.obs or adata.obs["plate"].isna().any()):
        raise ValueError("batch_unit=plate requires non-missing obs['plate'] for every crop")
    adata = adata.copy()
    experiment = adata.obs.get("experiment_batch", adata.obs["batch"]).astype(str)
    adata.obs["experiment_batch"] = experiment
    adata.obs["batch"] = (
        experiment + "|plate_" + adata.obs["plate"].astype(str) if unit == "plate" else experiment
    )
    return adata


def cell_type_folds(batches: np.ndarray, n_folds: int, rng, cell_type: str | None) -> list:
    """Inner tuning folds over batches. When pooling cell types, deal each type's batches
    round the folds. Experiment and plate labels both start with <cell type>-<nn>."""
    if cell_type is not None:
        return batch_folds(batches, n_folds, rng)
    dealt = sorted(rng.permutation(np.unique(batches)), key=lambda b: b.split("-")[0])
    return [np.array(dealt[k::n_folds]) for k in range(n_folds)]


def add_positive_controls_to_proxies(adata: sc.AnnData) -> sc.AnnData:
    """A copy of `adata` with the positive-control crops relabelled "control", joining the proxy W."""
    positive = _wells(adata, "positive_control")
    adata = adata.copy()
    adata.obs["perturbation"] = np.where(positive, "control", perturbation_labels(adata))
    return adata


def select_proxy_controls(adata: sc.AnnData, cfg: DictConfig) -> sc.AnnData:
    """Choose negative, positive or both control types without changing the input.

    In positive-only mode negative controls are excluded; in negative-only mode positive
    controls retain their siRNA labels as training perturbations.
    """
    mode = cfg.get("proxy_controls", "positive")
    if mode not in ("negative", "positive", "both"):
        raise ValueError("proxy_controls must be 'negative', 'positive' or 'both'")
    negative, positive = _wells(adata, "negative_control"), _wells(adata, "positive_control")
    proxies = {"negative": negative, "positive": positive, "both": negative | positive}[mode]
    batch = batch_labels(adata)
    missing = np.setdiff1d(batch, batch[proxies])
    if len(missing):
        raise ValueError(f"No {mode} proxy controls in batches: {', '.join(missing)}")
    if mode == "negative":
        return adata
    if mode == "positive":
        adata = adata[~negative]
    return add_positive_controls_to_proxies(adata)


def validate_treatment_overlap(train: sc.AnnData, val: sc.AnnData) -> None:
    """Require at least one treatment with a training label in every validation batch."""
    known = np.unique(perturbation_labels(train))
    scorable = _wells(val, "treatment") & np.isin(perturbation_labels(val), known)
    batch = batch_labels(val)
    missing = np.setdiff1d(batch, batch[scorable])
    if len(missing):
        raise ValueError(
            f"No treatment siRNAs shared with training for validation batches: {', '.join(missing)}. "
            "Use more training batches or different validation splits (seed/inner_folds)."
        )


def score_batch(models: dict, test: sc.AnnData) -> pd.DataFrame:
    """Scores of each model on one held-out batch: (perturbation, metric, model, value)."""
    labels = perturbation_labels(test)
    present = np.unique(labels[_wells(test, "treatment")])
    perts = np.intersect1d(present, next(iter(models.values())).perturbations_)
    X = as_dense(test.X)
    obs = np.stack([X[labels == p].mean(axis=0) for p in perts])
    rows = [
        _long(perts, name, "mse", mse(m.predict(perts, test), obs)) for name, m in models.items()
    ]
    return pd.concat(rows, ignore_index=True)


def score_batches(models: dict, adata: sc.AnnData, batches) -> pd.DataFrame:
    """score_batch on each of `batches` of `adata`, tagged with its batch."""
    batch = batch_labels(adata)
    return pd.concat(
        [score_batch(models, adata[batch == b]).assign(batch=b) for b in batches],
        ignore_index=True,
    )


def _wells(adata: sc.AnnData, well_type: str) -> np.ndarray:
    """Mask of the crops from wells of one type: treatment, negative_control or positive_control."""
    return (adata.obs["well_type"] == well_type).to_numpy()


def _long(perts: np.ndarray, model: str, metric: str, values: np.ndarray) -> pd.DataFrame:
    """One model's per-perturbation values of one metric as long-table rows."""
    return pd.DataFrame({"perturbation": perts, "metric": metric, "model": model, "value": values})


def cross_validate(adata: sc.AnnData, cfg: DictConfig) -> Iterator[pd.DataFrame]:
    """Hold out n_validation_batches distinct batches one at a time, selected using cfg.seed.

    Each model is fit on every other batch, including those held out in other runs.
    Yield each run's scores immediately; "fold" is the zero-based run index.
    """
    adata = select_batch_unit(adata, cfg.batch_unit)
    adata = select_proxy_controls(adata, cfg)
    print(f"proxy crops: {is_control(adata).sum():,} of {adata.n_obs:,}", flush=True)
    batch = batch_labels(adata)
    selected = validation_batches(batch, cfg.n_validation_batches, np.random.default_rng(cfg.seed))
    for i, held_out in enumerate(selected):
        print(f"validation {i + 1}/{len(selected)}: holding out {held_out}", flush=True)
        train = adata[batch != held_out]
        validate_treatment_overlap(train, adata[batch == held_out])
        fold_cfg = OmegaConf.merge(cfg, {"neural": {"log_key": f"fold {i + 1}/neural bridge"}})
        start = time.perf_counter()
        models = fit_models(train, fold_cfg)
        print(f"  fit {len(models)} models in {time.perf_counter() - start:.0f}s", flush=True)
        scores = score_batches(models, adata, [held_out]).assign(fold=i)
        scores["n_training_batches"] = scores["perturbation"].map(training_batch_counts(train))
        yield scores


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean per (model, metric): siRNAs averaged within each batch, then batches."""
    per_batch = scores.groupby(["metric", "model", "batch"])["value"].mean()
    return per_batch.groupby(["metric", "model"]).mean().unstack("metric")


def log_summary(prefix: str, table: pd.DataFrame) -> None:
    """Every value of a summary() table into the open W&B run's summary, as <prefix>/<model>/<metric>."""
    if wandb.run is not None:
        wandb.summary.update(
            {f"{prefix}/{m}/{metric}": v for (m, metric), v in table.stack().items()}
        )


@hydra.main(config_path="../configs", config_name="train_rxrx1_transfer", version_base=None)
def main(cfg: DictConfig):
    adata = select_cell_type(load_rxrx1(**cfg.embeddings), cfg.cell_type)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    folds = []
    config = OmegaConf.to_container(cfg, resolve=True)
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
    report_by_training_batches(pd.concat(folds), out_dir / "plots" / "by_training_batches")
    print(f"all folds:\n{overall.to_string(float_format='%.4g')}\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
