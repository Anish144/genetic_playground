"""Visual check of the linearity assumption in the batch-transfer model.

The model says, per gene, Y_e(x) = gamma_x + b * W_e. Removing each cell's perturbation effect
gamma_x and averaging per batch gives r_e = mean over the batch's perturbed cells of (y - gamma_x),
which the model predicts to be exactly b * W_e. This plots r_e against W_e (one dot per batch) with
the fitted line b * W_e and, dashed, a quadratic fit to the dots. If the dots follow the line with
no bend, linearity holds. The black line is the best straight line through the dots by
construction, so this is a visual check, not a test.

Genes are ranked by how much their control mean W_e varies across batches (std); one figure each
for the largest, middle and smallest spread. Fit on all batches (not held-out).
Config: configs/check_linearity.yaml. Usage:

    uv run python scripts/check_linearity.py n_genes=16

Saves outputs/<date>/<time>/plots/linearity_{high,middle,low}.png.
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


def plot_genes(genes, names, W, r, slope, cells, title: str, path: Path) -> None:
    """Grid of r_e vs W_e panels, one per gene, dots coloured by cells per batch."""
    ncols = int(np.ceil(np.sqrt(len(genes))))
    nrows = int(np.ceil(len(genes) / ncols))
    with plt.rc_context(bundles.icml2024(column="full", nrows=nrows, ncols=ncols, usetex=False)):
        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        for ax, g in zip(axes.flat, genes, strict=False):
            x, y = W[:, g], r[:, g]
            grid = np.linspace(x.min(), x.max(), 50)
            dots = ax.scatter(x, y, c=cells, cmap=CMAP, s=6, lw=0)
            ax.plot(grid, slope[g] * grid, color="#0b0b0b", lw=1)
            ax.plot(grid, np.polyval(np.polyfit(x, y, 2), grid), color="#eb6834", lw=1, ls="--")
            ax.set_title(f"{names[g]} (b = {slope[g]:.2f}, sd = {x.std():.2g})")
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(genes) :]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.supxlabel("control mean in batch, $W_e$")
        fig.supylabel(r"batch mean of $y - \gamma_x$")
        fig.colorbar(dots, ax=axes, label="perturbed cells per batch", shrink=0.6)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print(f"Saved {path}")


@hydra.main(config_path="../configs", config_name="check_linearity", version_base=None)
def main(cfg: DictConfig):
    adata = load_dataset()
    model = BatchTransferModel().fit(adata)
    s = summary_stats(adata)

    W = s["W"]  # (batches, genes)
    r = (s["batch_y_sum"] - s["pert_batch_n"].T @ model.gamma_) / s["batch_n"][:, None]
    spread = W.std(axis=0)
    ranked = np.argsort(-spread)[
        : (spread > 0).sum()
    ]  # largest spread first; constant genes dropped
    n, mid = cfg.n_genes, len(ranked) // 2
    groups = {
        "high": ranked[:n],
        "middle": ranked[mid - n // 2 : mid - n // 2 + n],
        "low": ranked[-n:],
    }

    plot_dir = Path(HydraConfig.get().runtime.output_dir) / "plots"
    for label, genes in groups.items():
        title = f"{label} spread of $W_e$ across batches"
        args = (adata.var_names, W, r, model.slope_, s["batch_n"])
        plot_genes(genes, *args, title, plot_dir / f"linearity_{label}.png")


if __name__ == "__main__":
    main()
