"""Visual check of the linearity assumption in the batch-transfer model.

The model says, per gene, Y_e(x) = gamma_x + b * W_e. Removing each cell's perturbation effect
gamma_x and averaging per batch gives r_e = mean over the batch's perturbed cells of (y - gamma_x),
which the model predicts to be exactly b * W_e. For the genes whose control mean W_e varies most
across batches, this plots r_e against W_e (one dot per batch) with the fitted line b * W_e and,
dashed, a quadratic fit to the dots. If the dots follow the line with no bend, linearity holds.

Fit on all batches (a diagnostic, not held-out). Config: configs/check_linearity.yaml. Usage:

    uv run python scripts/check_linearity.py n_genes=16

Saves outputs/<date>/<time>/plots/linearity.png.
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tueplots import bundles

from genetic_perturbation_playground.data.data_loader import load_dataset
from genetic_perturbation_playground.model.batch_transfer import BatchTransferModel, summary_stats
from genetic_perturbation_playground.utils.plotting import CMAP


@hydra.main(config_path="../configs", config_name="check_linearity", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    model = BatchTransferModel().fit(adata)
    s = summary_stats(adata)

    W = s["W"]  # (batches, genes)
    r = (s["batch_y_sum"] - s["pert_batch_n"].T @ model.gamma_) / s["batch_n"][:, None]
    genes = np.argsort(-W.std(axis=0))[: cfg.n_genes]

    ncols = int(np.ceil(np.sqrt(cfg.n_genes)))
    nrows = int(np.ceil(cfg.n_genes / ncols))
    with plt.rc_context(bundles.icml2024(column="full", nrows=nrows, ncols=ncols, usetex=False)):
        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        for ax, g in zip(axes.flat, genes, strict=False):
            x, y = W[:, g], r[:, g]
            grid = np.linspace(x.min(), x.max(), 50)
            dots = ax.scatter(x, y, c=s["batch_n"], cmap=CMAP, s=6, lw=0)
            ax.plot(grid, model.slope_[g] * grid, color="#0b0b0b", lw=1)
            ax.plot(grid, np.polyval(np.polyfit(x, y, 2), grid), color="#eb6834", lw=1, ls="--")
            ax.set_title(f"{adata.var_names[g]} (b = {model.slope_[g]:.2f})")
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(genes) :]:
            ax.set_visible(False)
        fig.supxlabel("control mean in batch, $W_e$")
        fig.supylabel(r"batch mean of $y - \gamma_x$")
        fig.colorbar(dots, ax=axes, label="perturbed cells per batch", shrink=0.6)

        out = Path(HydraConfig.get().runtime.output_dir) / "plots" / "linearity.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
