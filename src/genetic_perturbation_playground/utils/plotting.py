from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            lo, hi = ax.get_ylim()
            if hi - lo >= 1:  # one tick per decade; narrower ranges keep the default ticks
                ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{10**t:.2g}"))
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


def report(scores: pd.DataFrame, unit: str, out_dir: Path) -> None:
    """Print each model's mean score and save one plot per metric to out_dir.

    `scores` is a long table with columns [unit, metric, model, value, cells].
    """
    for name, m in scores.groupby("metric", sort=False):
        wide = m.pivot(index=unit, columns="model", values="value")[m["model"].unique()]
        cells = m.groupby(unit)["cells"].first().reindex(wide.index).to_numpy()
        for label in wide:
            print(f"{unit:>12} | {name:>13} | {label:<18}: {wide[label].mean():.4f}")
        series = {label: wide[label].to_numpy() for label in wide}
        plot_metric(series, name, out_dir / f"{name}.png", cells, log=name == "mse", unit=unit)
    print(f"Plots saved to {out_dir}")


def plot_mse_vs_n_envs(
    scores: pd.DataFrame, models: dict[str, str], floor: str | None, path: Path
) -> None:
    """Mean ± sd of MSE against the number of source environments, one panel per theta*.

    `scores` is a long table with columns [n_envs, theta_star, model, mse]. `models` maps the
    model names to plot onto legend labels, in the fixed categorical colour order; `floor`
    names a model drawn as a grey reference line (e.g. the irreducible error).
    """
    colors = ["#2a78d6", "#eb6834"]  # categorical slots 1-2, validated light-mode palette
    stats = scores.groupby(["theta_star", "model", "n_envs"])["mse"].agg(["mean", "std"])
    thetas = scores["theta_star"].unique()
    with plt.rc_context(bundles.icml2024(column="full", ncols=len(thetas), usetex=False)):
        fig, axes = plt.subplots(1, len(thetas), squeeze=False, sharey=True)
        for ax, theta in zip(axes.flat, thetas, strict=True):
            if floor is not None:
                ref = stats.loc[(theta, floor), "mean"]
                ax.plot(ref.index, ref, color="#8a8984", lw=1, label="irreducible error")
            for (model, label), color in zip(models.items(), colors, strict=False):
                s = stats.loc[(theta, model)]
                ax.fill_between(
                    s.index,
                    s["mean"] - s["std"],
                    s["mean"] + s["std"],
                    color=color,
                    alpha=0.1,
                    lw=0,
                )
                ax.plot(
                    s.index,
                    s["mean"],
                    color=color,
                    lw=2,
                    marker="o",
                    ms=5,
                    mec="white",
                    mew=1.5,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xticks(s.index, [str(d) for d in s.index])
            ax.minorticks_off()
            ax.set(title=rf"$\theta^\star = {theta:g}$", xlabel="number of source environments")
            ax.grid(color="#e5e4df", lw=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes.flat[0].set_ylabel("held-out target MSE")
        axes.flat[0].legend(frameon=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print(f"Saved {path}")
