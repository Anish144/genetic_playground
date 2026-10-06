"""Train the linear perturbation model on Replogle K562 and report held-out metrics.

Cells are split at random into train/test. The model is fit on train; the "true" effects are
the plain (alpha=0) mean shifts on the test cells. Config: configs/train_linear.yaml. Usage:

    uv run python scripts/train_linear.py alpha=10

Plots (one per metric) are written to the Hydra run dir: outputs/<date>/<time>/plots/.
"""

from pathlib import Path

import hydra
import numpy as np
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.linear import LinearPerturbationModel
from genetic_perturbation_playground.utils import metrics
from genetic_perturbation_playground.utils.plotting import plot_metric


def evaluate(adata: sc.AnnData, alpha: float, test_frac: float, seed: int, k: int) -> dict:
    """Per-perturbation metrics: {metric: {series: array of shape (n_perts,)}}."""
    is_test = np.random.default_rng(seed).random(adata.n_obs) < test_frac
    model = LinearPerturbationModel(alpha=alpha).fit(adata[~is_test])
    truth = LinearPerturbationModel(alpha=0.0).fit(adata[is_test])

    perts = np.intersect1d(model.perturbations_, truth.perturbations_)
    pred, true = model.predict(perts), truth.predict(perts)
    zero = np.zeros_like(true)  # "no effect" baseline for reference

    return {
        "mse": {"linear": metrics.mse(pred, true), "no-effect baseline": metrics.mse(zero, true)},
        "pearson": {"linear": metrics.pearson_per_row(pred, true)},
        f"pearson_top{k}": {"linear": metrics.pearson_top_k(pred, true, k)},
    }


@hydra.main(config_path="../configs", config_name="train_linear", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset(cfg.dataset)
    results = evaluate(adata, cfg.alpha, cfg.test_frac, cfg.seed, cfg.top_k)

    plot_dir = Path(HydraConfig.get().runtime.output_dir) / "plots"
    for name, series in results.items():
        for label, values in series.items():
            print(f"{name:>12} | {label:<20}: {np.nanmean(values):.4f}")
        plot_metric(series, name, plot_dir / f"{name}.png", log=name == "mse")
    print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
