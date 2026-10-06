"""Train the linear perturbation model on Replogle K562 and report held-out metrics.

Cells are split at random into train/test. The model is fit on train; the "true" effects are
the plain (alpha=0) mean shifts on the test cells. Config: configs/train_linear.yaml. Usage:

    uv run python scripts/train_linear.py alpha=10
"""

import hydra
import numpy as np
import scanpy as sc
from omegaconf import DictConfig

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.linear import LinearPerturbationModel
from genetic_perturbation_playground.utils import metrics


def evaluate(adata: sc.AnnData, alpha: float, test_frac: float, seed: int, k: int) -> dict:
    is_test = np.random.default_rng(seed).random(adata.n_obs) < test_frac
    model = LinearPerturbationModel(alpha=alpha).fit(adata[~is_test])
    truth = LinearPerturbationModel(alpha=0.0).fit(adata[is_test])

    perts = np.intersect1d(model.perturbations_, truth.perturbations_)
    pred, true = model.predict(perts), truth.predict(perts)
    zero = np.zeros_like(true)  # "no effect" baseline for reference

    return {
        "n_perturbations": len(perts),
        "mse": metrics.mse(pred, true),
        "mse_zero_baseline": metrics.mse(zero, true),
        "pearson": np.nanmean(metrics.pearson_per_row(pred, true)),
        f"pearson_top{k}": np.nanmean(metrics.pearson_top_k(pred, true, k)),
    }


@hydra.main(config_path="../configs", config_name="train_linear", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset(cfg.dataset)
    results = evaluate(adata, cfg.alpha, cfg.test_frac, cfg.seed, cfg.top_k)
    for name, value in results.items():
        print(f"{name:>22}: {value:.4f}" if isinstance(value, float) else f"{name:>22}: {value}")


if __name__ == "__main__":
    main()
