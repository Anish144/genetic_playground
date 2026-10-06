from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from tueplots import bundles

STYLE = bundles.icml2024(column="half", usetex=False)
CMAP = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#3987e5", "#0d366b"])  # light→dark


def plot_metric(
    values: dict[str, np.ndarray],
    name: str,
    path: Path,
    cells: np.ndarray,
    log: bool = False,
    unit: str = "perturbation",
) -> None:
    """Violin + jittered dots of a metric scored per `unit`; one violin per entry in `values`.

    Dots are coloured by `cells`, the number of cells per unit (same order as values).
    With log=True the violin is fit to log10 values, so skewed metrics (e.g. mse) stay readable.
    """
    rng = np.random.default_rng(0)
    n = len(cells)
    dot_size = 1.5 if n > 500 else 4
    norm = LogNorm(cells.min(), cells.max())

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ticklabels = []
        for i, (label, v) in enumerate(values.items()):
            keep = np.isfinite(v) & (v > 0) if log else np.isfinite(v)
            v, c = v[keep], cells[keep]
            y, y_mean = (np.log10(v), np.log10(v.mean())) if log else (v, v.mean())
            parts = ax.violinplot(y, positions=[i], widths=0.8, showextrema=False)
            for body in parts["bodies"]:
                body.set(facecolor="#e5e4df", edgecolor="#8a8984", lw=0.6, alpha=1)
            order = np.argsort(c)  # draw well-sampled units on top
            jitter = rng.uniform(-0.15, 0.15, len(y))
            dots = ax.scatter(
                (i + jitter)[order], y[order], c=c[order], cmap=CMAP, norm=norm, s=dot_size, lw=0
            )
            ax.hlines(y_mean, i - 0.3, i + 0.3, color="#0b0b0b", lw=1)
            ticklabels.append(f"{label}\nmean {v.mean():.3g}")

        if log:
            ax.yaxis.set_major_locator(MultipleLocator(1))  # one tick per decade
            ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"$10^{{{t:g}}}$"))
        ax.set(xticks=range(len(values)), xticklabels=ticklabels, ylabel=name)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_title(f"{name} per {unit} (n = {n})")
        ax.grid(axis="y", color="#e5e4df", lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        cbar = fig.colorbar(dots, ax=ax, label=f"cells per {unit}", pad=0.02)
        cbar.ax.minorticks_off()
        cbar.set_ticks(np.unique(np.geomspace(cells.min(), cells.max(), 4).round()))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}"))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)


def report(
    results: dict[str, dict[str, np.ndarray]],
    cells: np.ndarray,
    out_dir: Path,
    unit: str = "perturbation",
) -> None:
    """Print the mean of each metric (scored per `unit`) and save one plot per metric to out_dir."""
    for name, series in results.items():
        for label, values in series.items():
            print(f"{unit:>12} | {name:>12} | {label:<20}: {np.nanmean(values):.4f}")
        plot_metric(series, name, out_dir / f"{name}.png", cells, log=name == "mse", unit=unit)
    print(f"Plots saved to {out_dir}")
