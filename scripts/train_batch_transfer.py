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


def score(models: dict, test: sc.AnnData, k: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-perturbation metrics {metric: {model: (n_perts,)}} on `test`, cells per perturbation,
    and the perturbation names.

    Truth is each perturbation's mean minus its batch-matched control mean in `test`.
    """
    s = summary_stats(test)
    keep = np.isin(s["perturbations"], next(iter(models.values())).perturbations_)
    perts, w = s["perturbations"][keep], s["w_mean"][keep]
    true = s["y_mean"][keep] - w
    preds = {name: m.predict(perts, w) - w for name, m in models.items()}
    results = {
        "mse": {
            **{n: metrics.mse(p, true) for n, p in preds.items()},
            "no effect": metrics.mse(np.zeros_like(true), true),
        },
        "pearson": {n: metrics.pearson_per_row(p, true) for n, p in preds.items()},
        f"pearson_top{k}": {n: metrics.pearson_top_k(p, true, k) for n, p in preds.items()},
    }
    return results, s["n"][keep], perts


def score_per_batch(models: dict, test: sc.AnnData, k: int) -> tuple[dict, np.ndarray]:
    """One score per held-out batch (mean over its perturbations), and cells per batch."""
    batch = test.obs["batch"].astype(str).to_numpy()
    batches = np.unique(batch)
    per_batch = [score(models, test[batch == b], k)[0] for b in batches]
    results = {
        metric: {
            model: np.array([np.nanmean(r[metric][model]) for r in per_batch]) for model in series
        }
        for metric, series in per_batch[0].items()
    }
    return results, np.array([(batch == b).sum() for b in batches])


def cross_validate(adata: sc.AnnData, n_folds: int, seed: int, k: int) -> tuple:
    """K-fold over batches: each batch is held out exactly once, models refit per fold.

    Returns (per-perturbation results, cells per perturbation, per-batch results, cells per batch).
    Per-perturbation scores are averaged over folds (cells summed); per-batch scores are stacked.
    """
    batch = adata.obs["batch"].astype(str).to_numpy()
    folds = np.array_split(np.random.default_rng(seed).permutation(np.unique(batch)), n_folds)

    pert_frames, batch_scores = [], []
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{n_folds}: holding out {len(held_out)} batches")
        is_test = np.isin(batch, held_out)
        models, test = fit(adata[~is_test]), adata[is_test]
        results, cells, perts = score(models, test, k)
        pert_frames.append(
            pd.DataFrame(
                {(m, s): v for m, ser in results.items() for s, v in ser.items()}, index=perts
            ).assign(cells=cells)
        )
        batch_scores.append(score_per_batch(models, test, k))

    by_pert = pd.concat(pert_frames).groupby(level=0)
    pert_df = by_pert.mean().drop(columns="cells")
    pert_results = {
        m: {s: pert_df[(m, s)].to_numpy() for s in series} for m, series in results.items()
    }
    batch_results = {
        m: {s: np.concatenate([r[m][s] for r, _ in batch_scores]) for s in series}
        for m, series in results.items()
    }
    batch_cells = np.concatenate([c for _, c in batch_scores])
    return pert_results, by_pert["cells"].sum().to_numpy(), batch_results, batch_cells


@hydra.main(config_path="../configs", config_name="train_batch_transfer", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    pert_results, pert_cells, batch_results, batch_cells = cross_validate(
        adata, cfg.n_folds, cfg.seed, cfg.top_k
    )

    plot_dir = Path(HydraConfig.get().runtime.output_dir) / "plots"
    report(pert_results, pert_cells, plot_dir / "per_perturbation")
    report(batch_results, batch_cells, plot_dir / "per_batch", unit="batch")


if __name__ == "__main__":
    main()
