"""Out-of-batch prediction on Replogle K562 with the batch-transfer model.

Whole batches are held out. Models are fit on the training batches; in the held-out batches only
the control cells are used (W_{e*}) to predict each perturbation's mean expression. Metrics compare
predicted vs observed effects (perturbation mean - control mean in the held-out batches).
Config: configs/train_batch_transfer.yaml. Usage:

    uv run python scripts/train_batch_transfer.py test_batch_frac=0.3

Plots (one per metric) are written to the Hydra run dir: outputs/<date>/<time>/plots/.
"""

from pathlib import Path

import hydra
import numpy as np
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.batch_transfer import BatchTransferModel, summary_stats
from genetic_perturbation_playground.utils import metrics
from genetic_perturbation_playground.utils.plotting import report

MODELS = {"pooled (b=0)": 0.0, "shift (b=1)": 1.0, "transfer (fit b)": None}


def evaluate(
    adata: sc.AnnData, test_batch_frac: float, seed: int, k: int
) -> tuple[dict, np.ndarray]:
    """Per-perturbation metrics {metric: {model: (n_perts,)}} and test cells per perturbation."""
    batch = adata.obs["batch"].astype(str).to_numpy()
    batches = np.unique(batch)
    n_test = max(1, round(test_batch_frac * len(batches)))
    test_batches = np.random.default_rng(seed).permutation(batches)[:n_test]
    is_test = np.isin(batch, test_batches)

    models = {
        name: BatchTransferModel(slope).fit(adata[~is_test]) for name, slope in MODELS.items()
    }
    test = summary_stats(adata[is_test])
    keep = np.isin(test["perturbations"], models["pooled (b=0)"].perturbations_)
    perts, w = test["perturbations"][keep], test["w_mean"][keep]
    true = test["y_mean"][keep] - w  # observed effect vs the held-out batches' controls

    slope = models["transfer (fit b)"].slope_
    print(
        f"held out {n_test}/{len(batches)} batches | fitted b: median {np.median(slope):.3f}, "
        f"IQR [{np.percentile(slope, 25):.3f}, {np.percentile(slope, 75):.3f}]"
    )

    preds = {name: m.predict(perts, w) - w for name, m in models.items()}
    results = {
        "mse": {
            **{n: metrics.mse(p, true) for n, p in preds.items()},
            "no effect": metrics.mse(np.zeros_like(true), true),
        },
        "pearson": {n: metrics.pearson_per_row(p, true) for n, p in preds.items()},
        f"pearson_top{k}": {n: metrics.pearson_top_k(p, true, k) for n, p in preds.items()},
    }
    return results, test["n"][keep]


@hydra.main(config_path="../configs", config_name="train_batch_transfer", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    results, cells = evaluate(adata, cfg.test_batch_frac, cfg.seed, cfg.top_k)

    report(results, cells, Path(HydraConfig.get().runtime.output_dir) / "plots")


if __name__ == "__main__":
    main()
