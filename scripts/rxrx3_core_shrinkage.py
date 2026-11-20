"""The control shift with empirical-Bayes per-plate weights (rxrx3_core_proxy.shrunk_shift) on one
released set, against no shift, the full shift and half the shift, after each official alignment.

    uv run python scripts/rxrx3_core_shrinkage.py <set>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rxrx3_core_proxy import CORE, ROOT, SETS, control_shift, efaar, recall, shrinkage_weights, shrunk_shift, wells  # noqa: E402
from efaar_benchmarking.efaar import pca_centerscale_on_controls, tvn_on_controls  # noqa: E402

NAME = sys.argv[1]


def main():
    meta = pd.read_csv(CORE / "metadata_rxrx3_core.csv", low_memory=False)
    meta["perturbation"] = meta["treatment"].where(meta["treatment"].str.contains("_control"), meta["treatment"].str.split("_").str[0])
    meta["plate_batch"] = meta["experiment_name"] + "/" + meta["plate"].astype(str)
    X, obs = wells(CORE / SETS[NAME], meta)
    out = []
    for base, align in {
        "PCA-CS": lambda: pca_centerscale_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
        "TVN": lambda: tvn_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
        "EFAAR": lambda: efaar(X, obs),
    }.items():
        Y = align()
        alpha = shrinkage_weights(Y, obs)
        print(f"{NAME} {base}: alpha quantiles 0/10/50/90/100% {np.round(alpha.quantile([0, .1, .5, .9, 1]).to_numpy(), 3)}, "
              f"lowest on {alpha.idxmin()}", flush=True)
        for label, Z in {
            "none": Y, "full shift": control_shift(Y, obs), "alpha 1": shrunk_shift(Y, obs, 1.0),
            "alpha 0.5": shrunk_shift(Y, obs, 0.5), "empirical Bayes": shrunk_shift(Y, obs),
        }.items():
            out.append(recall(Z, obs).assign(embedding=NAME, map=base, shift=label).reset_index())
    df = pd.concat(out)
    df.to_csv(ROOT / f"outputs/2026-11-15/rxrx3_core_shrinkage_{NAME}.csv", index=False)
    for metric in ["both 0.95", "both 0.99", "both 0.995", "both 0.999"]:
        print(f"\n{metric}\n", df.pivot_table(index=["map", "shift"], columns="source", values=metric, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
