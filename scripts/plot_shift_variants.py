"""Recall gain of each control-shift variant over no shift, from rxrx3_core_shift_diagnostics.py.

Rows: databases; columns: both-tail cutoffs; within a panel, one line per released set and map.

    uv run python scripts/plot_shift_variants.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parents[1] / "outputs/2026-11-15"
SETS = {"released": "OpenPhenom", "Phenom-1": "Phenom-1", "Phenom-2": "Phenom-2", "CellProfiler": "CellProfiler"}
MAPS = ["TVN", "EFAAR"]
DBS = ["CORUM", "HuMAP", "StringDB", "Reactome", "SIGNOR"]
CUTOFFS = {"both 0.95": "95th", "both 0.99": "99th", "both 0.995": "99.5th", "both 0.999": "99.9th"}
VARIANTS = {  # label, colour (dataviz categorical slots 1-3), marker
    "shift": ("full shift", "#2a78d6", "o"),
    "half": ("half shift", "#eb6834", "s"),
    "median": ("median shift", "#1baf7a", "D"),
}


def load() -> pd.DataFrame:
    frames = []
    for key, name in SETS.items():
        for m in MAPS:
            d = pd.read_csv(OUT / f"shift_diagnostics_{key}_{m}.csv", header=[0, 1], index_col=0)
            frames.append(d.stack([0, 1], future_stack=True).rename("recall").rename_axis(["variant", "db", "cutoff"])
                          .reset_index().assign(row=f"{name} {m}"))
    return pd.concat(frames)


def main():
    df = load().pivot_table(index=["row", "db", "cutoff"], columns="variant", values="recall")
    rows = [f"{s} {m}" for s in SETS.values() for m in MAPS]
    fig, axes = plt.subplots(len(DBS), len(CUTOFFS), figsize=(15, 16), sharey=True, sharex="col")
    for i, db in enumerate(DBS):
        for j, cut in enumerate(CUTOFFS):
            ax = axes[i, j]
            d = df.xs((db, cut), level=["db", "cutoff"]).reindex(rows)
            for y in range(len(rows)):
                ax.axhline(y, color="#e6e5e0", lw=0.8, zorder=0)
            ax.axvline(0, color="#6b6a64", lw=1, zorder=1)
            for k, (v, (label, colour, marker)) in enumerate(VARIANTS.items()):
                ax.scatter(d[v] - d["none"], [y + (k - 1) * 0.22 for y in range(len(rows))], s=36, color=colour,
                           marker=marker, edgecolor="white", linewidth=1, zorder=3, label=label)
            ax.set_yticks(range(len(rows)), rows)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9, colors="#3d3c38")
            if i == 0:
                ax.set_title(f"{CUTOFFS[cut]} percentile", fontsize=11, color="#1f1e1c")
            if j == 0:
                ax.set_ylabel(db, fontsize=12, color="#1f1e1c", rotation=0, ha="right", va="center", labelpad=90)
            if i == len(DBS) - 1:
                ax.set_xlabel("recall minus no-shift recall", fontsize=9, color="#3d3c38")
    axes[0, 0].invert_yaxis()  # shared: first set at the top of every panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.55, 1.0))
    fig.suptitle("Control-shift variants on RxRx3-core: gain in both-tail recall over no shift", y=1.02, fontsize=14, color="#1f1e1c")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "shift_variants.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
