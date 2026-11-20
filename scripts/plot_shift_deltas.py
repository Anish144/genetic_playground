"""Recall for each proxy shift (full, half, empirical Bayes), one figure per alignment: rows
databases, columns percentile cut-offs (both tails), bars grouped by released set, no shift as a
dotted line over each group. The differences from no shift go to shift_deltas.csv.
Reads the rxrx3_core_shrinkage_<set>.csv files.

    uv run python scripts/plot_shift_deltas.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parents[1] / "outputs/2026-11-15"
SETS = {"released": "OpenPhenom", "Phenom-1": "Phenom-1", "Phenom-2": "Phenom-2", "CellProfiler": "CellProfiler"}
SHIFTS = {"full shift": ("full shift", "#2a78d6"), "alpha 0.5": ("half shift", "#eb6834"), "empirical Bayes": ("empirical Bayes", "#1baf7a")}
DBS = ["CORUM", "HuMAP", "Reactome", "StringDB"]
THRS = ["both 0.95", "both 0.99", "both 0.995", "both 0.999"]


def main():
    df = pd.concat(pd.read_csv(OUT / f"rxrx3_core_shrinkage_{s}.csv") for s in SETS)
    df["embedding"] = df["embedding"].map(SETS)
    long = df.melt(id_vars=["source", "embedding", "map", "shift"], value_vars=THRS, var_name="thr")
    wide = long.pivot_table(index=["map", "source", "thr", "embedding"], columns="shift", values="value")
    delta = wide[list(SHIFTS)].sub(wide["none"], axis=0)
    delta.to_csv(OUT / "shift_deltas.csv")
    x, width = np.arange(len(SETS)), 0.26
    for base in ["PCA-CS", "TVN"]:
        fig, axes = plt.subplots(len(DBS), len(THRS), figsize=(15, 11), sharex=True)
        for (i, db), (j, thr) in ((a, b) for a in enumerate(DBS) for b in enumerate(THRS)):
            ax, d = axes[i, j], wide.loc[(base, db, thr)].reindex(SETS.values())
            for k, (col, (label, color)) in enumerate(SHIFTS.items()):
                ax.bar(x + (k - 1) * width, d[col], width * 0.9, color=color, label=label)
            ax.hlines(d["none"], x - 1.6 * width, x + 1.6 * width, color="#222", ls=":", lw=1.6, label="no shift")
            ax.grid(axis="y", color="#ddd", lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(x, SETS.values(), rotation=30, ha="right", fontsize=9)
            if i == 0:
                ax.set_title(f"{float(thr.split()[1]) * 100:g}th percentile (both tails)")
            if j == 0:
                ax.set_ylabel(f"{db}\nrecall")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"RxRx3-core, {base}: known-relationship recall by shift", y=1.03, fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT / f"shift_deltas_{base}.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
