"""Proxy methods vs the official alignments on RxRx3-core, scored with the official
EFAAR_benchmarking known_relationship_benchmark at several percentiles.

Maps (well embeddings -> alignment -> mean per gene over its CRISPR wells):
  PCA-CS experiment     official pca_centerscale_on_controls on EMPTY_control, batch = experiment
  TVN experiment        official tvn_on_controls, batch = experiment
  EFAAR                 efaar below: per plate, standardise, PCA, standardise, TVN (eps 0.5)
  <map> + control shift minus, per plate, the mean of its EMPTY_control and positive-control wells
                        (gene plates: ~12 'CRISPR Gene Positive Controls' guide wells, e.g. PLK1, MTOR, SRC), then
                        minus the negatives' mean, so only plate-to-plate differences of the shift remain
                        (the official scorer is raw cosine; a shared offset would swamp it)

Recall: the official compute_recall with threshold pairs (l, r). Both tails (1-p, p); similar tail
only (0, p); opposite tail only (1-p, 1) -- rank 0 / rank 1 can't be reached, so those sides are off.

    uv run python scripts/rxrx3_core_proxy.py [set names] [--exclude-positives]    (~190 GB RAM, 16 CPUs, a few hours)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rxrx3_core_repro import CORE, ROOT, wells  # noqa: E402  (also stubs the official package's unused imports)

from efaar_benchmarking.benchmarking import known_relationship_benchmark  # noqa: E402
from efaar_benchmarking.efaar import pca_centerscale_on_controls, tvn_on_controls  # noqa: E402
from sklearn.utils import Bunch  # noqa: E402

SETS = {
    "ours": "ours_openphenom.parquet", "released": "OpenPhenom_rxrx3_core_embeddings.parquet",
    "CellProfiler": "CellProfiler_features_rxrx3_core.parquet",
    "Phenom-1": "MAE-L_8_RPI-93M_rxrx3_core_embeddings.parquet",
    "Phenom-2": "MAE-G_8_PP-16M_trimmed_rxrx3_core_embeddings.parquet",
}
NAMES = [a for a in sys.argv[1:] if not a.startswith("--")] or ["ours", "released"]  # set names to run
POSITIVE = ["CRISPR Gene Positive Controls"]  # PLK1, MTOR, SRC... guides
EXCLUDE_POSITIVES = "--exclude-positives" in sys.argv  # drop the positive-control genes from the scored set
PCTS = [0.95, 0.99, 0.995, 0.999]
PAIRS = {f"both {p:g}": (round(1 - p, 4), p) for p in PCTS} | {f"sim {p:g}": (0.0, p) for p in PCTS} | {"opp 0.95": (0.05, 1.0)}


def recall(X: np.ndarray, obs: pd.DataFrame) -> pd.DataFrame:
    crispr = (obs["perturbation_type"] == "CRISPR").to_numpy() & ~obs["perturbation"].str.contains("_control").to_numpy()
    if EXCLUDE_POSITIVES:
        crispr &= ~obs["perturbation"].isin(obs.loc[obs["well_type_label"].isin(POSITIVE), "perturbation"]).to_numpy()
    genes = pd.DataFrame(X[crispr]).groupby(obs["perturbation"].to_numpy()[crispr]).mean()
    r = known_relationship_benchmark(
        Bunch(metadata=pd.DataFrame({"gene": genes.index}), features=genes.reset_index(drop=True)),
        pert_col="gene", recall_thr_pairs=list(PAIRS.values()),
    ).set_index("source")
    return pd.DataFrame({k: r[f"recall_{min(v)}_{max(v)}"] for k, v in PAIRS.items()})


def control_shift(X: np.ndarray, obs: pd.DataFrame) -> np.ndarray:
    X, plates = X.copy(), obs["plate_batch"].to_numpy()
    controls = ((obs["perturbation"] == "EMPTY_control") | obs["well_type_label"].isin(POSITIVE)).to_numpy()
    for p in np.unique(plates):
        X[plates == p] -= X[(plates == p) & controls].mean(axis=0)
    return X - X[(obs["perturbation"] == "EMPTY_control").to_numpy()].mean(axis=0)  # negatives back at 0, as gene_map does


def shrinkage_weights(X: np.ndarray, obs: pd.DataFrame) -> pd.Series:
    """Empirical-Bayes weight per plate, alpha_p = tau^2 / (tau^2 + v_p), for the shift m_p - mbar.

    m_p is the plate's control mean (EMPTY + positive wells); v_p = RSS_p / ((n_p - G_p) n_p) its
    sampling variance (trace), RSS_p the squared deviations of the plate's control wells from their
    group (EMPTY / positive) means and G_p the number of groups; tau^2 the
    between-plate variance of the true offsets, mean ||m_p - mbar||^2 - mean v_p (positive part).
    Plates with and without positive wells are pooled separately, as their m_p measure different things.
    """
    plates = obs["plate_batch"].to_numpy()
    groups = {"negative": (obs["perturbation"] == "EMPTY_control").to_numpy(), "positive": obs["well_type_label"].isin(POSITIVE).to_numpy()}
    rows = []
    for p in np.unique(plates):
        Ys = [X[(plates == p) & g] for g in groups.values() if ((plates == p) & g).any()]
        n = sum(len(Y) for Y in Ys)
        rows.append({"plate": p, "has_positive": len(Ys) == 2, "m": np.concatenate(Ys).mean(0),
                     "v": sum(((Y - Y.mean(0)) ** 2).sum() for Y in Ys) / ((n - len(Ys)) * n)})
    df = pd.DataFrame(rows).set_index("plate")
    alpha = pd.Series(index=df.index, dtype=float)
    for _, d in df.groupby("has_positive"):
        M = np.stack(d["m"])
        tau2 = max(((M - M.mean(0)) ** 2).sum(1).mean() - d["v"].mean(), 0.0)
        alpha[d.index] = tau2 / (tau2 + d["v"])
    return alpha


def shrunk_shift(X: np.ndarray, obs: pd.DataFrame, alpha: float | None = None) -> np.ndarray:
    """y - alpha_p (m_p - mbar), mbar the mean control mean over plates of the same kind (with or
    without positive wells), then re-centred on the negatives. alpha None: empirical-Bayes weights."""
    plates = obs["plate_batch"].to_numpy()
    controls = ((obs["perturbation"] == "EMPTY_control") | obs["well_type_label"].isin(POSITIVE)).to_numpy()
    has_positive = pd.Series(obs["well_type_label"].isin(POSITIVE).to_numpy()).groupby(plates).any()
    weights = shrinkage_weights(X, obs) if alpha is None else pd.Series(alpha, index=has_positive.index)
    m = {p: X[(plates == p) & controls].mean(0) for p in has_positive.index}
    mbar = {k: np.mean([m[p] for p in has_positive.index[has_positive == k]], axis=0) for k in (True, False)}
    X = X.copy()
    for p in has_positive.index:
        X[plates == p] -= weights[p] * (m[p] - mbar[has_positive[p]])
    return X - X[(obs["perturbation"] == "EMPTY_control").to_numpy()].mean(axis=0)


def efaar(X: np.ndarray, obs: pd.DataFrame, eps: float = 0.5) -> np.ndarray:
    """Per plate p, x -> (x - m_p) D1_p V D2_p (C_p + eps I)^-1/2 (C + eps I)^1/2, all fit on EMPTY_control wells:
    m_p, D1_p the plate's mean and 1/std; V the PCA rotation of every standardised control; D2_p the
    plate's 1/std in that basis; C_p the plate's covariance after D2_p and C that of every control."""
    plates, negative = obs["plate_batch"].to_numpy(), (obs["perturbation"] == "EMPTY_control").to_numpy()
    names = np.unique(plates)
    controls = {p: X[negative & (plates == p)] for p in names}
    m = {p: c.mean(axis=0) for p, c in controls.items()}
    D1 = {p: 1 / c.std(axis=0) for p, c in controls.items()}
    _, V = np.linalg.eigh(np.cov(np.concatenate([(controls[p] - m[p]) * D1[p] for p in names]), rowvar=False))
    V = V[:, ::-1]
    Z = {p: (controls[p] - m[p]) * D1[p] @ V for p in names}
    D2 = {p: 1 / z.std(axis=0) for p, z in Z.items()}
    Z = {p: Z[p] * D2[p] for p in names}

    def power(z, q):
        vals, vecs = np.linalg.eigh(np.cov(z, rowvar=False) + eps * np.eye(z.shape[1]))
        return (vecs * vals**q) @ vecs.T

    target, out = power(np.concatenate(list(Z.values())), 0.5), np.empty_like(X)
    for p in names:
        out[plates == p] = ((X[plates == p] - m[p]) * D1[p] @ V * D2[p]) @ power(Z[p], -0.5) @ target
    return out


def main():
    meta = pd.read_csv(CORE / "metadata_rxrx3_core.csv", low_memory=False)
    meta["perturbation"] = meta["treatment"].where(meta["treatment"].str.contains("_control"), meta["treatment"].str.split("_").str[0])
    meta["plate_batch"] = meta["experiment_name"] + "/" + meta["plate"].astype(str)
    out = []
    for name in NAMES:
        X, obs = wells(CORE / SETS[name], meta)
        print("positive-control genes:", sorted(obs.loc[obs["well_type_label"].isin(POSITIVE), "perturbation"].unique()), flush=True)
        base = {
            "PCA-CS experiment": pca_centerscale_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
            "TVN experiment": tvn_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
            "EFAAR": efaar(X, obs),
        }
        for how, Y in base.items():
            for label, Z in {how: Y, f"{how} + control shift": control_shift(Y, obs)}.items():
                r = recall(Z, obs).assign(embedding=name, map=label)
                print(name, label, r.loc[["CORUM", "HuMAP", "StringDB"], ["both 0.95", "sim 0.999"]].round(3).to_dict(), flush=True)
                out.append(r.reset_index())
    df = pd.concat(out)
    df.to_csv(ROOT / f"outputs/2026-11-15/rxrx3_core_proxy{''.join('_' + n for n in NAMES if n not in ('ours', 'released'))}{'_nopos' if EXCLUDE_POSITIVES else ''}.csv", index=False)
    for metric in PAIRS:
        print(f"\n{metric}\n", df.pivot_table(index=["embedding", "map"], columns="source", values=metric, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
