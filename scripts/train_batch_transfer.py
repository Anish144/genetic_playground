"""Out-of-batch prediction on Replogle K562 (or RxRx1 Phenom embeddings) with the batch-transfer model.

K-fold cross-validation over batches: the batches are split into n_folds groups; each group is
held out once while the models are fit on the rest. In held-out batches only the control cells are
used (W_{e*}) to predict each perturbation's mean expression. Metrics compare predicted vs observed
effects (perturbation mean - batch-matched control mean). Two views:

- per perturbation: within a fold the held-out batches are pooled, one score per perturbation;
  averaged over folds                                                    -> plots/per_perturbation/
- per batch: each batch scored on its own when held out (mean of the per-perturbation scores
  within that batch); every batch appears once                          -> plots/per_batch/

Raw scores (long tables) go to scores_per_perturbation.csv and scores_per_batch.csv.

Config: configs/train_batch_transfer.yaml. `name` is required and names the output folder. Usage:

    uv run python scripts/train_batch_transfer.py name=baseline n_folds=10
    uv run python scripts/train_batch_transfer.py name=kernel_only 'bridges=[kernel]'   # no GPU: skip the neural bridge
    uv run python scripts/train_batch_transfer.py name=neural_500 neural.n_steps=500 neural.weight_decay=1e-2
    uv run python scripts/train_batch_transfer.py name=rxrx1 dataset=rxrx1   # run scripts/embed_rxrx1.py first
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.data.rxrx1 import load_rxrx1
from genetic_perturbation_playground.model.batch_transfer import (
    BatchTransferModel,
    KernelBridgeTransfer,
    NeuralBridgeTransfer,
    batch_labels,
    is_control,
    summary_stats,
)
from genetic_perturbation_playground.utils import metrics
from genetic_perturbation_playground.utils.plotting import report

MODELS = {"pooled (b=0)": 0.0, "shift (b=1)": 1.0, "transfer (fit b)": None}


def fit(train: sc.AnnData, cfg: DictConfig) -> dict:
    """Fit every model variant on the training batches."""
    models = _fit_linear(train) | _fit_bridges(train, cfg)
    _print_slope(models["transfer (fit b)"].slope_)
    return models


def _fit_linear(train: sc.AnnData) -> dict:
    """The three BatchTransferModel variants of MODELS."""
    return {name: BatchTransferModel(slope).fit(train) for name, slope in MODELS.items()}


def _fit_bridges(train: sc.AnnData, cfg: DictConfig) -> dict:
    """The bridges named in cfg.bridges."""
    return {f"{b} bridge": _make_bridge(b, cfg).fit(train) for b in cfg.bridges}


def _make_bridge(kind: str, cfg: DictConfig):
    """An unfitted kernel or neural bridge with the config's settings."""
    if kind == "kernel":
        return KernelBridgeTransfer(cfg.bridge_reg, cfg.weight_by_cells)
    return NeuralBridgeTransfer(
        cfg.weight_by_cells, reg=cfg.bridge_reg, seed=cfg.seed, **cfg.neural
    )


def _print_slope(slope: np.ndarray) -> None:
    """Median and interquartile range of the fitted per-gene slopes."""
    q25, median, q75 = np.percentile(slope, [25, 50, 75])
    print(f"fitted b: median {median:.3f}, IQR [{q25:.3f}, {q75:.3f}]")


def score(models: dict, test: sc.AnnData, k: int, seen: np.ndarray) -> pd.DataFrame:
    """Per-perturbation scores on `test` (long table, see metrics.score) plus cells per perturbation.

    Effects are relative to the batch-matched control mean in `test`. The control cells are split:
    the models see only those marked `seen` (the proxy), and both predicted and true effects
    subtract the mean of the rest, so the proxy's controls are never reused as the baseline.
    """
    inputs, baseline = _split_controls(test, seen)
    s = summary_stats(baseline)
    keep = np.isin(s["perturbations"], next(iter(models.values())).perturbations_)
    perts = s["perturbations"][keep]
    w = s["w_mean"][keep]
    preds = {name: m.predict(perts, inputs) - w for name, m in models.items()}
    scores = metrics.score(preds, s["y_mean"][keep] - w, perts, k)
    cells = dict(zip(perts, s["n"][keep], strict=True))
    return scores.assign(cells=scores["perturbation"].map(cells))


def _split_controls(test: sc.AnnData, seen: np.ndarray) -> tuple:
    """`test` with only the seen controls (model inputs), and with only the rest (baseline)."""
    ctrl = is_control(test)
    return test[~ctrl | seen], test[~ctrl | ~seen]


def cross_validate(adata: sc.AnnData, cfg: DictConfig) -> tuple:
    """K-fold over batches: each batch is held out exactly once, models refit per fold.

    Every batch's controls are split in two halves. The `seen` half is the proxy for all models, in
    training and held-out batches alike; the other half is only the baseline in scoring.

    Returns (per-perturbation scores, per-batch scores) as long tables. Per-perturbation scores are
    averaged over folds (cells summed); per-batch scores average over the batch's perturbations.
    """
    batch = batch_labels(adata)
    rng = np.random.default_rng(cfg.seed)
    folds = _batch_folds(batch, cfg.n_folds, rng)
    seen = _seen_controls(adata, batch, rng)
    train_cells = ~is_control(adata) | seen

    pert_scores, batch_scores = [], []
    for i, held_out in enumerate(folds):
        print(f"fold {i + 1}/{cfg.n_folds}: holding out {len(held_out)} batches")
        is_test = np.isin(batch, held_out)
        models = fit(adata[~is_test & train_cells], cfg)
        test = adata[is_test]
        pert_scores.append(score(models, test, cfg.top_k, seen[is_test]))
        _print_fold_means(pert_scores[-1])
        batch_scores += _score_batches(models, test, seen[is_test], held_out, batch, cfg.top_k)
    return _average_per_perturbation(pert_scores), _average_per_batch(batch_scores)


def _batch_folds(batch: np.ndarray, n_folds: int, rng: np.random.Generator) -> list:
    """The batches, shuffled and split into n_folds groups."""
    return np.array_split(rng.permutation(np.unique(batch)), n_folds)


def _seen_controls(adata: sc.AnnData, batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mask of the controls the models see: every other one within each batch, shuffled."""
    rank = pd.Series(rng.random(adata.n_obs)).groupby([batch, is_control(adata)]).rank()
    return rank.to_numpy() % 2 == 0


def _score_batches(models, test, seen, held_out, batch, k) -> list[pd.DataFrame]:
    """Scores of each held-out batch on its own, tagged with the batch and its total cells."""
    test_batch = batch_labels(test)
    scores = []
    for b in held_out:
        in_b = test_batch == b
        s = score(models, test[in_b], k, seen[in_b])
        scores.append(s.assign(batch=b, cells=(batch == b).sum()))
    return scores


def _print_fold_means(scores: pd.DataFrame) -> None:
    """Each model's mean score per metric in one fold."""
    means = scores.groupby(["metric", "model"], sort=False)["value"].mean()
    print(means.unstack("model").to_string(float_format="%.4g"), flush=True)


def _average_per_perturbation(scores: list[pd.DataFrame]) -> pd.DataFrame:
    """Scores averaged over folds per perturbation, with cells summed."""
    grouped = pd.concat(scores).groupby(["perturbation", "metric", "model"], sort=False)
    return grouped.agg(value=("value", "mean"), cells=("cells", "sum")).reset_index()


def _average_per_batch(scores: list[pd.DataFrame]) -> pd.DataFrame:
    """Scores averaged over each batch's perturbations."""
    grouped = pd.concat(scores).groupby(["batch", "metric", "model"], sort=False)
    return grouped.agg(value=("value", "mean"), cells=("cells", "first")).reset_index()


@hydra.main(config_path="../configs", config_name="train_batch_transfer", version_base=None)
def main(cfg: DictConfig):
    adata = load_data(cfg)
    per_pert, per_batch = cross_validate(adata, cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    _save_scores(per_pert, per_batch, out_dir)
    _plot_scores(per_pert, per_batch, out_dir / "plots")


def load_data(cfg: DictConfig) -> sc.AnnData:
    """The dataset named by cfg.dataset: Replogle cells x genes, or RxRx1 crops x Phenom features."""
    return {"replogle": load_dataset, "rxrx1": load_rxrx1}[cfg.dataset]()


def _save_scores(per_pert: pd.DataFrame, per_batch: pd.DataFrame, out_dir: Path) -> None:
    """Both score tables as CSVs in out_dir."""
    per_pert.to_csv(out_dir / "scores_per_perturbation.csv", index=False)
    per_batch.to_csv(out_dir / "scores_per_batch.csv", index=False)


def _plot_scores(per_pert: pd.DataFrame, per_batch: pd.DataFrame, plot_dir: Path) -> None:
    """One plot per metric for each view, printing the model means."""
    report(per_pert, "perturbation", plot_dir / "per_perturbation")
    report(per_batch, "batch", plot_dir / "per_batch")


if __name__ == "__main__":
    main()
