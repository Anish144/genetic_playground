"""Gene-gene recall on RxRx3-core from the public embeddings, following
EFAAR_benchmarking/RxRx3-core_benchmarks/rxrx3_core_benchmarks_openphenom.ipynb step for step.
The results are compared with Table 6 of arXiv 2503.20158.

The only additions to the notebook are:
  - a loop over the four released embedding sets;
  - TVN (efaar.tvn_on_controls) alongside the notebook's PCA-CS;
  - dropping NaN and constant features, which only CellProfiler has.

    pip install "efaar_benchmarking @ git+https://github.com/recursionpharma/EFAAR_benchmarking@2935f21" huggingface_hub pyarrow
    python rxrx3_core_minimal_repro.py
"""

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
PAPER = {  # arXiv 2503.20158 Table 6 (current version), RxRx3-core (PCA-CS), recall at (0.05, 0.95)
    "CellProfiler": {"CORUM": .361, "HuMAP": .444, "Reactome": .160, "StringDB": .330},
    "OpenPhenom": {"CORUM": .300, "HuMAP": .352, "Reactome": .158, "StringDB": .281},
    "Phenom-1": {"CORUM": .395, "HuMAP": .482, "Reactome": .188, "StringDB": .349},
    "Phenom-2": {"CORUM": .486, "HuMAP": .553, "Reactome": .197, "StringDB": .415},
}


def download(filename):
    return hf_hub_download("recursionpharma/rxrx3-core", filename=filename, repo_type="dataset")


def recall(features, metadata):
    """The notebook's map building and gene-gene benchmark, on aligned well features."""
    aligned = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(features).add_prefix("feature_")], axis=1)
    aligned = aligned[~((aligned["perturbation_type"] == "COMPOUND") & aligned["perturbation"].str.contains("_control"))]
    feature_cols = [c for c in aligned.columns if c.startswith("feature_")]
    map_data = aligned.groupby(["perturbation_type", "perturbation", "concentration"], dropna=False)[feature_cols].mean().reset_index()
    map_data = map_data[map_data["concentration"].isin(COMPOUND_CONCENTRATIONS) | map_data["concentration"].isna()]
    genes = map_data.query("perturbation_type == 'CRISPR'")
    result = known_relationship_benchmark(
        Bunch(metadata=genes.drop(columns=feature_cols), features=genes[feature_cols]),
        recall_thr_pairs=[(0.05, 0.95)], pert_col="perturbation",
    )
    return result.set_index("source")["recall_0.05_0.95"]


def main():
    metadata = pd.read_csv(download("metadata_rxrx3_core.csv"), low_memory=False)
    metadata["perturbation"] = metadata["treatment"].apply(lambda x: x.split("_")[0] if "_control" not in x else x)
    rows = []
    for name, filename in EMBEDDINGS.items():
        embeddings = pd.read_parquet(download(filename)).groupby("well_id").mean()
        merged = metadata.merge(embeddings, left_on="well_id", right_index=True)
        X = merged[embeddings.columns].to_numpy(float)
        X = X[:, np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)]
        meta = merged[metadata.columns]
        for alignment, aligner in {"PCA-CS": pca_centerscale_on_controls, "TVN": tvn_on_controls}.items():
            aligned = aligner(X, meta, pert_col="perturbation", control_key="EMPTY_control", batch_col="experiment_name")
            rows.append(recall(aligned, meta).rename((name, alignment)))
        rows.append(pd.Series(PAPER[name], name=(name, "paper Table 6")))
        print(f"{name} done", flush=True)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
