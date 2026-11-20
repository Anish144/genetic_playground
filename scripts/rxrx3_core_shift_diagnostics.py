"""Why does the control shift hurt Phenom-2 TVN and break CellProfiler EFAAR?

For one set and base map: the size of the per-plate shift against the gene signal, how concentrated it
is in a few features, and recall under shift variants (negatives only, positives only, median, half).

    uv run python scripts/rxrx3_core_shift_diagnostics.py <set> <TVN|EFAAR|PCA-CS>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rxrx3_core_proxy import CORE, POSITIVE, ROOT, SETS, efaar, recall, wells  # noqa: E402
from efaar_benchmarking.efaar import pca_centerscale_on_controls, tvn_on_controls  # noqa: E402

NAME, BASE = sys.argv[1], sys.argv[2]


def plate_shift(X, obs, mask, stat=np.mean, alpha=1.0):
    """Minus alpha * the per-plate statistic of the masked wells, then recentred on the negatives."""
    X, plates = X.copy(), obs["plate_batch"].to_numpy()
    for p in np.unique(plates[mask]):  # plates without masked wells are not shifted
        X[plates == p] -= alpha * stat(X[(plates == p) & mask], axis=0)
    return X - X[(obs["perturbation"] == "EMPTY_control").to_numpy()].mean(axis=0)


def main():
    meta = pd.read_csv(CORE / "metadata_rxrx3_core.csv", low_memory=False)
    meta["perturbation"] = meta["treatment"].where(meta["treatment"].str.contains("_control"), meta["treatment"].str.split("_").str[0])
    meta["plate_batch"] = meta["experiment_name"] + "/" + meta["plate"].astype(str)
    X, obs = wells(CORE / SETS[NAME], meta)
    Y = {
        "PCA-CS": lambda: pca_centerscale_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
        "TVN": lambda: tvn_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
        "EFAAR": lambda: efaar(X, obs),
    }[BASE]()

    neg = (obs["perturbation"] == "EMPTY_control").to_numpy()
    pos = obs["well_type_label"].isin(POSITIVE).to_numpy()
    gene = (obs["perturbation_type"] == "CRISPR").to_numpy() & ~obs["perturbation"].str.contains("_control").to_numpy() & ~pos
    plates = obs["plate_batch"].to_numpy()
    gene_plates = np.unique(plates[pos])

    # size of the shift against the signal, on gene plates
    shifts = np.stack([Y[(plates == p) & (neg | pos)].mean(0) for p in gene_plates])
    shifts -= Y[neg].mean(0)
    centred = Y - Y[neg].mean(0)
    pos_effect = np.stack([Y[(plates == p) & pos].mean(0) - Y[(plates == p) & neg].mean(0) for p in gene_plates])
    sq = np.sort(shifts**2, axis=1)[:, ::-1]
    print(f"{NAME} {BASE}: {Y.shape[1]} features, {len(gene_plates)} gene plates, positive wells/plate "
          f"{np.median([(pos & (plates == p)).sum() for p in gene_plates]):.0f}")
    print(f"  median |gene well - neg mean|       {np.median(np.linalg.norm(centred[gene], axis=1)):.3g}")
    print(f"  median |negative well - neg mean|   {np.median(np.linalg.norm(centred[neg], axis=1)):.3g}")
    print(f"  median |plate shift|                {np.median(np.linalg.norm(shifts, axis=1)):.3g}  (max {np.linalg.norm(shifts, axis=1).max():.3g})")
    print(f"  median |positive - negative| /plate {np.median(np.linalg.norm(pos_effect, axis=1)):.3g}")
    print(f"  cosine of plate positive effects to their mean {np.median(pos_effect @ pos_effect.mean(0) / np.linalg.norm(pos_effect, axis=1) / np.linalg.norm(pos_effect.mean(0))):.3f}")
    worst = gene_plates[np.argmax(np.linalg.norm(shifts, axis=1))]
    print(f"  largest shift on {worst}: features {np.argsort(-np.abs(shifts[np.argmax(np.linalg.norm(shifts, axis=1))]))[:5]}, "
          f"plates with |shift| > 10x median: {(np.linalg.norm(shifts, axis=1) > 10 * np.median(np.linalg.norm(shifts, axis=1))).sum()}")
    print(f"  share of |shift|^2 in top 10 features {np.median(sq[:, :10].sum(1) / sq.sum(1)):.3f}", flush=True)

    variants = {
        "none": Y,
        "shift": plate_shift(Y, obs, neg | pos),
        "negatives only": plate_shift(Y, obs, neg),
        "positives only": plate_shift(Y, obs, pos),
        "median": plate_shift(Y, obs, neg | pos, np.median),
        "half": plate_shift(Y, obs, neg | pos, alpha=0.5),
    }
    rows = []
    for label, Z in variants.items():
        r = recall(Z, obs)[["both 0.95", "both 0.99", "both 0.995", "both 0.999"]]
        rows.append(r.stack().rename(label))
        print(label, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"outputs/2026-11-15/shift_diagnostics_{NAME}_{BASE}.csv")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
