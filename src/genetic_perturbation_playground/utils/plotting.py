from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

STYLE = bundles.icml2024(column="half", usetex=False)
COLORS = ["#2a78d6", "#8a8984"]  # model (blue), reference baseline (gray)


def plot_metric(values: dict[str, np.ndarray], name: str, path: Path) -> None:
    """Violin + jittered dots of a per-perturbation metric; one violin per entry in `values`."""
    rng = np.random.default_rng(0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for i, (color, v) in enumerate(zip(COLORS, values.values(), strict=False)):
            v = v[np.isfinite(v)]
            parts = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False)
            for body in parts["bodies"]:
                body.set(facecolor=color, edgecolor="none", alpha=0.25)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=2, color=color, alpha=0.5, lw=0)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color="#0b0b0b", lw=1)
            ax.annotate(f"mean {v.mean():.3f}", (i + 0.32, v.mean()), va="center", fontsize=6)

        n = len(next(iter(values.values())))
        ax.set(xticks=range(len(values)), xticklabels=list(values), ylabel=name)
        ax.set_xlim(-0.6, len(values) - 0.1)
        ax.set_title(f"{name} per perturbation (n = {n})")
        ax.grid(axis="y", color="#e5e4df", lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)
