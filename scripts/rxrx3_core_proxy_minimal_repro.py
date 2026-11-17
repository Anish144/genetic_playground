"""Gene-gene recall on RxRx3-core with and without the per-plate control shift, from the public
embeddings. Map building and scoring are as in rxrx3_core_minimal_repro.py (the official
EFAAR_benchmarking notebook, PCA-CS and TVN on EMPTY_control by experiment).

The control shift, applied to the aligned well features before averaging wells into genes:
  m_p    mean of plate p's control wells (EMPTY_control and the ~12 'CRISPR Gene Positive Controls')
  m̄      mean of m_p over plates of the same kind (with / without positive-control wells)
  y  <-  y - alpha_p (m_p - m̄)   for every well on plate p, then the EMPTY_control mean is
         subtracted from all wells (the scorer uses raw cosine, so a shared offset would swamp it).
alpha_p is 0 (no shift), 1 (full shift), 0.5 (half shift), or the empirical-Bayes weight
  alpha_p = tau^2 / (tau^2 + v_p),
  v_p    sampling variance of m_p (trace): squared deviations of the control wells from their
         group (EMPTY / positive) means, / ((n_p - G_p) n_p), G_p the number of groups on the plate
  tau^2  variance of the true plate offsets: mean ||m_p - m̄||^2 - mean v_p over plates of the kind.

    pip install "efaar_benchmarking @ git+https://github.com/recursionpharma/EFAAR_benchmarking@2935f21" huggingface_hub pyarrow matplotlib
    python rxrx3_core_proxy_minimal_repro.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from efaar_benchmarking.benchmarking import known_relationship_benchmark
from efaar_benchmarking.constants import COMPOUND_CONCENTRATIONS
from efaar_benchmarking.efaar import pca_centerscale_on_controls, tvn_on_controls
from huggingface_hub import hf_hub_download
from sklearn.utils import Bunch

EMBEDDINGS = {
    "CellProfiler": "CellProfiler_features_rxrx3_core.parquet",
    "OpenPhenom": "OpenPhenom_rxrx3_core_embeddings.parquet",
    "Phenom-1": "MAE-L_8_RPI-93M_rxrx3_core_embeddings.parquet",
    "Phenom-2": "MAE-G_8_PP-16M_trimmed_rxrx3_core_embeddings.parquet",
}
PAPER = {  # arXiv 2503.20158 Table 6, RxRx3-core (PCA-CS), recall at (0.05, 0.95)
    "CellProfiler": {"CORUM": .361, "HuMAP": .444, "Reactome": .160, "StringDB": .330},
    "OpenPhenom": {"CORUM": .300, "HuMAP": .352, "Reactome": .158, "StringDB": .281},
    "Phenom-1": {"CORUM": .395, "HuMAP": .482, "Reactome": .188, "StringDB": .349},
    "Phenom-2": {"CORUM": .486, "HuMAP": .553, "Reactome": .197, "StringDB": .415},
}
ALIGNERS = {"PCA-CS": pca_centerscale_on_controls, "TVN": tvn_on_controls}
SHIFTS = {"no shift": 0.0, "full shift": 1.0, "half shift": 0.5, "empirical Bayes": None}
PERCENTILES = [0.95, 0.99, 0.995, 0.999]
DATABASES = ["CORUM", "HuMAP", "Reactome", "StringDB"]


def download(filename):
    return hf_hub_download("recursionpharma/rxrx3-core", filename=filename, repo_type="dataset")


def recall(features, metadata):
    """The notebook's map building and gene-gene benchmark, both tails at each percentile."""
    aligned = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(features).add_prefix("feature_")], axis=1)
    aligned = aligned[~((aligned["perturbation_type"] == "COMPOUND") & aligned["perturbation"].str.contains("_control"))]
    feature_cols = [c for c in aligned.columns if c.startswith("feature_")]
    map_data = aligned.groupby(["perturbation_type", "perturbation", "concentration"], dropna=False)[feature_cols].mean().reset_index()
    map_data = map_data[map_data["concentration"].isin(COMPOUND_CONCENTRATIONS) | map_data["concentration"].isna()]
    genes = map_data.query("perturbation_type == 'CRISPR'")
    result = known_relationship_benchmark(
        Bunch(metadata=genes.drop(columns=feature_cols), features=genes[feature_cols]),
        recall_thr_pairs=[(round(1 - p, 4), p) for p in PERCENTILES], pert_col="perturbation",
    ).set_index("source")
    return pd.DataFrame({p: result[f"recall_{round(1 - p, 4)}_{p}"] for p in PERCENTILES})


def plate_controls(X, metadata):
    """Per plate: whether it has positive-control wells, the control mean m and its variance v."""
    negative = (metadata["perturbation"] == "EMPTY_control").to_numpy()
    positive = (metadata["well_type_label"] == "CRISPR Gene Positive Controls").to_numpy()
    rows = {}
    for plate in metadata["plate_batch"].unique():
        on_plate = (metadata["plate_batch"] == plate).to_numpy()
        groups = [X[on_plate & g] for g in (negative, positive) if (on_plate & g).any()]
        n = sum(len(g) for g in groups)
        rss = sum(((g - g.mean(axis=0)) ** 2).sum() for g in groups)
        rows[plate] = {"has_positive": len(groups) == 2, "m": np.concatenate(groups).mean(axis=0), "v": rss / ((n - len(groups)) * n)}
    plates = pd.DataFrame.from_dict(rows, orient="index")
    mbar = {kind: np.mean(np.stack(group["m"]), axis=0) for kind, group in plates.groupby("has_positive")}
    plates["mbar"] = [mbar[kind] for kind in plates["has_positive"]]
    return plates


def empirical_bayes(plates):
    """alpha_p = tau^2 / (tau^2 + v_p), tau^2 estimated separately for each kind of plate."""
    alpha = pd.Series(index=plates.index, dtype=float)
    for _, kind in plates.groupby("has_positive"):
        spread = np.mean([((m - mbar) ** 2).sum() for m, mbar in zip(kind["m"], kind["mbar"])])
        tau2 = max(spread - kind["v"].mean(), 0.0)
        alpha[kind.index] = tau2 / (tau2 + kind["v"])
    return alpha


def control_shift(X, metadata, alpha):
    """y - alpha_p (m_p - m̄) on every plate, then the negatives' mean subtracted. alpha None: empirical Bayes."""
    plates = plate_controls(X, metadata)
    alpha = empirical_bayes(plates) if alpha is None else pd.Series(alpha, index=plates.index)
    offsets = np.stack([a * (m - mbar) for a, m, mbar in zip(alpha[plates.index], plates["m"], plates["mbar"])])
    shifted = X - offsets[plates.index.get_indexer(metadata["plate_batch"])]
    return shifted - shifted[(metadata["perturbation"] == "EMPTY_control").to_numpy()].mean(axis=0)


def plot(results, alignment):
    fig, axes = plt.subplots(len(DATABASES), len(PERCENTILES), figsize=(15, 11), sharex=True)
    x, width, colors = np.arange(len(EMBEDDINGS)), 0.26, ["#2a78d6", "#eb6834", "#1baf7a"]
    for i, database in enumerate(DATABASES):
        for j, percentile in enumerate(PERCENTILES):
            ax = axes[i, j]
            r = results.xs((alignment, database), level=("alignment", "database"))[percentile].unstack("shift").loc[list(EMBEDDINGS)]
            for k, (shift, color) in enumerate(zip(list(SHIFTS)[1:], colors)):
                ax.bar(x + (k - 1) * width, r[shift], width * 0.9, color=color, label=shift)
            ax.hlines(r["no shift"], x - 1.6 * width, x + 1.6 * width, color="#222", ls=":", lw=1.6, label="no shift")
            ax.grid(axis="y", color="#ddd", lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(x, list(EMBEDDINGS), rotation=30, ha="right", fontsize=9)
            if i == 0:
                ax.set_title(f"{percentile * 100:g}th percentile (both tails)")
            if j == 0:
                ax.set_ylabel(f"{database}\nrecall")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"RxRx3-core, {alignment}: known-relationship recall by shift", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"rxrx3_core_shift_{alignment}.png", dpi=130, bbox_inches="tight")


def main():
    metadata = pd.read_csv(download("metadata_rxrx3_core.csv"), low_memory=False)
    metadata["perturbation"] = metadata["treatment"].apply(lambda x: x.split("_")[0] if "_control" not in x else x)
    metadata["plate_batch"] = metadata["experiment_name"] + "/" + metadata["plate"].astype(str)
    results = {}
    for name, filename in EMBEDDINGS.items():
        embeddings = pd.read_parquet(download(filename)).groupby("well_id").mean()
        merged = metadata.merge(embeddings, left_on="well_id", right_index=True)
        X = merged[embeddings.columns].to_numpy(float)
        X = X[:, np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)]  # CellProfiler has NaN / constant features
        meta = merged[metadata.columns].reset_index(drop=True)
        for alignment, aligner in ALIGNERS.items():
            aligned = aligner(X, meta, pert_col="perturbation", control_key="EMPTY_control", batch_col="experiment_name")
            for shift, alpha in SHIFTS.items():
                results[(name, alignment, shift)] = recall(control_shift(aligned, meta, alpha), meta)
        print(f"{name} done", flush=True)
    results = pd.concat(results, names=["embedding", "alignment", "shift", "database"])
    results = results[results.index.get_level_values("database").isin(DATABASES)]
    results.to_csv("rxrx3_core_shift_recall.csv")
    for percentile in PERCENTILES:
        print(f"\nrecall, both tails at {percentile}\n", results[percentile].unstack("database").round(3).to_string())
    paper = pd.DataFrame(PAPER).T[DATABASES]
    print("\npaper Table 6 (PCA-CS, 0.95), for the no-shift rows\n", paper.to_string())
    for alignment in ALIGNERS:
        plot(results, alignment)


if __name__ == "__main__":
    main()
