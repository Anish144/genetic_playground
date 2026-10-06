"""Out-of-batch prediction on Replogle K562 with the batch-transfer model.

K-fold cross-validation over batches: the batches are split into n_folds groups; each group is
held out once while the models are fit on the rest. In held-out batches only the control cells are
used (W_{e*}) to predict each perturbation's mean expression. Metrics compare predicted vs observed
effects (perturbation mean - batch-matched control mean). Two views:

- per perturbation: within a fold the held-out batches are pooled, one score per perturbation;
  averaged over folds                                                    -> plots/per_perturbation/
- per batch: each batch scored on its own when held out (mean of the per-perturbation scores
  within that batch); every batch appears once                          -> plots/per_batch/

Config: configs/train_batch_transfer.yaml. Usage:

    uv run python scripts/train_batch_transfer.py n_folds=10
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.batch_transfer import BatchTransferModel, summary_stats
from genetic_perturbation_playground.utils import metrics
from genetic_perturbation_playground.utils.plotting import report

MODELS = {"pooled (b=0)": 0.0, "shift (b=1)": 1.0, "transfer (fit b)": None}


def fit(train: sc.AnnData) -> dict:
    """Fit every model variant on the training batches."""
    models = {name: BatchTransferModel(s).fit(train) for name, s in MODELS.items()}
    slope = models["transfer (fit b)"].slope_
    print(
        f"fitted b: median {np.median(slope):.3f}, "
        f"IQR [{np.percentile(slope, 25):.3f}, {np.percentile(slope, 75):.3f}]"
    )
    return models


def score(models: dict, test: sc.AnnData, k: int) -> pd.DataFrame:
    """Per-perturbation scores on `test` (long table, see metrics.score) plus cells per perturbation.

    Truth is each perturbation's mean minus its batch-matched control mean in `test`.
    """
    s = summary_stats(test)
    keep = np.isin(s["perturbations"], next(iter(models.values())).perturbations_)
    perts, w = s["perturbations"][keep], s["w_mean"][keep]
    preds = {name: m.predict(perts, w) - w for name, m in models.items()}
    scores = metrics.score(preds, s["y_mean"][keep] - w, perts, k)
    return scores.assign(
        cells=scores["perturbation"].map(dict(zip(perts, s["n"][keep], strict=True)))
    )


def cross_validate(adata: sc.AnnData, n_folds: int, seed: int, k: int) -> tuple:
    """K-fold over batches: each batch is held out exactly once, models refit per fold.

    Returns (per-perturbation scores, per-batch scores) as long tables. Per-perturbation scores are
    averaged over folds (cells summed); per-batch scores average over the batch's perturbations.
    """
    batch = adata.obs["batch"].astype(str).to_numpy()
    folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(batch)), n_folds)

    pert_scores, batch_scores = [], []
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{n_folds}: holding out {len(held_out)} batches")
        is_test = np.isin(batch, held_out)
        models, test = fit(adata[~is_test]), adata[is_test]
        pert_scores.append(score(models, test, k))
        batch_scores += [
            score(models, test[batch[is_test] == b], k).assign(batch=b, cells=(batch == b).sum())
            for b in held_out
        ]

    keys = ["metric", "model"]
    per_pert = pd.concat(pert_scores).groupby(["perturbation", *keys], sort=False)
    per_batch = pd.concat(batch_scores).groupby(["batch", *keys], sort=False)
    return (
        per_pert.agg(value=("value", "mean"), cells=("cells", "sum")).reset_index(),
        per_batch.agg(value=("value", "mean"), cells=("cells", "first")).reset_index(),
    )


@hydra.main(config_path="../configs", config_name="train_batch_transfer", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    per_pert, per_batch = cross_validate(adata, cfg.n_folds, cfg.seed, cfg.top_k)

    plot_dir = Path(HydraConfig.get().runtime.output_dir) / "plots"
    report(per_pert, "perturbation", plot_dir / "per_perturbation")
    report(per_batch, "batch", plot_dir / "per_batch")


if __name__ == "__main__":
    main()
