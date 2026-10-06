"""Train the linear perturbation model on Replogle K562 and report held-out metrics.

Cells are split at random into train/test. The model is fit on train; the "true" effects are
the same mean shifts computed on the test cells. Config: configs/train_linear.yaml. Usage:

    uv run python scripts/train_linear.py test_frac=0.1

Plots (one per metric) are written to the Hydra run dir: outputs/<date>/<time>/plots/.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.linear import LinearPerturbationModel
from genetic_perturbation_playground.utils import metrics
from genetic_perturbation_playground.utils.plotting import report


def evaluate(adata: sc.AnnData, test_frac: float, seed: int, k: int) -> pd.DataFrame:
    """Per-perturbation scores (long table, see metrics.score) plus cells per perturbation."""
    is_test = np.random.default_rng(seed).random(adata.n_obs) < test_frac
    model = LinearPerturbationModel().fit(adata[~is_test])
    truth = LinearPerturbationModel().fit(adata[is_test])

    perts = np.intersect1d(model.perturbations_, truth.perturbations_)
    scores = metrics.score({"linear": model.predict(perts)}, truth.predict(perts), perts, k)
    cells = adata.obs["perturbation"].astype(str).value_counts()
    return scores.assign(cells=scores["perturbation"].map(cells))


@hydra.main(config_path="../configs", config_name="train_linear", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    scores = evaluate(adata, cfg.test_frac, cfg.seed, cfg.top_k)
    report(scores, "perturbation", Path(HydraConfig.get().runtime.output_dir) / "plots")


if __name__ == "__main__":
    main()
