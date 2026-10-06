from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from tueplots import bundles

STYLE = bundles.icml2024(column="half", usetex=False)
COLORS = ["#2a78d6", "#8a8984"]  # model (blue), reference baseline (gray)


def plot_metric(values: dict[str, np.ndarray], name: str, path: Path, log: bool = False) -> None:
    """Violin + jittered dots of a per-perturbation metric; one violin per entry in `values`.

    With log=True the violin is fit to log10 values, so skewed metrics (e.g. mse) stay readable.
    """
    rng = np.random.default_rng(0)
    n = len(next(iter(values.values())))
    dot_size, dot_alpha = (1, 0.15) if n > 500 else (3, 0.5)  # keep the violin visible when dense

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for i, (color, v) in enumerate(zip(COLORS, values.values(), strict=False)):
            v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
            y, y_mean = (np.log10(v), np.log10(v.mean())) if log else (v, v.mean())
            parts = ax.violinplot(y, positions=[i], widths=0.8, showextrema=False)
            for body in parts["bodies"]:
                body.set(facecolor=color, edgecolor=color, lw=0.6, alpha=0.3)
            jitter = rng.uniform(-0.15, 0.15, len(y))
            ax.scatter(i + jitter, y, s=dot_size, color=color, alpha=dot_alpha, lw=0)
            ax.hlines(y_mean, i - 0.3, i + 0.3, color="#0b0b0b", lw=1)
            ax.annotate(f"mean {v.mean():.3g}", (i + 0.32, y_mean), va="center", fontsize=6)

        if log:
            ax.yaxis.set_major_locator(MultipleLocator(1))  # one tick per decade
            ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"$10^{{{t:g}}}$"))
        ax.set(xticks=range(len(values)), xticklabels=list(values), ylabel=name)
        ax.set_xlim(-0.6, len(values) - 0.1)
        ax.set_title(f"{name} per perturbation (n = {n})")
        ax.grid(axis="y", color="#e5e4df", lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)
