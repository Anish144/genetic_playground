"""Reproduce the RxRx3-core gene-gene recall of arXiv 2503.20158 (Table 6) with the official
EFAAR_benchmarking code, on every embedding set released on Hugging Face and on our own.

Each map: well embeddings (mean over a well's rows) -> alignment on EMPTY_control -> mean per gene
over its CRISPR wells -> official known_relationship_benchmark, recall_0.05_0.95.

Alignments (official efaar.py functions):
  PCA-CS experiment   pca_centerscale_on_controls, batch = experiment (the official notebook)
  PCA-CS plate        same, batch = experiment/plate
  TVN experiment      tvn_on_controls, batch = experiment
  none                raw embeddings

    uv run python scripts/rxrx3_core_repro.py [extra.parquet ...]    (~190 GB RAM, 16 CPUs)
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
for name in ["geomloss", "scvi", "scvi.model"]:  # only used by benchmarks we don't run
    sys.modules[name] = types.ModuleType(name)
sys.modules["geomloss"].SamplesLoss = sys.modules["scvi.model"].SCVI = None
sys.path.insert(0, str(ROOT / "outputs/scratch/EFAAR_benchmarking"))

from efaar_benchmarking.benchmarking import known_relationship_benchmark  # noqa: E402
from efaar_benchmarking.efaar import pca_centerscale_on_controls, tvn_on_controls  # noqa: E402
from sklearn.utils import Bunch  # noqa: E402

CORE = ROOT / "outputs/scratch/rxrx3_core"  # the Hugging Face dataset recursionpharma/rxrx3-core
(ROOT / "outputs/2026-11-15").mkdir(parents=True, exist_ok=True)  # where the RxRx3-core scripts write
PAPER = pd.DataFrame(  # arXiv 2503.20158 Table 6 (current version), PCA-CS
    {"CORUM": [.361, .300, .395, .486], "HuMAP": [.444, .352, .482, .553], "Reactome": [.160, .158, .188, .197], "StringDB": [.330, .281, .349, .415]},
    index=["CellProfiler", "OpenPhenom", "Phenom-1", "Phenom-2"],
)
RELEASED = {
    "CellProfiler": "CellProfiler_features_rxrx3_core.parquet",
    "OpenPhenom": "OpenPhenom_rxrx3_core_embeddings.parquet",
    "Phenom-1": "MAE-L_8_RPI-93M_rxrx3_core_embeddings.parquet",
    "Phenom-2": "MAE-G_8_PP-16M_trimmed_rxrx3_core_embeddings.parquet",
}


def wells(path: Path, meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    emb = pd.read_parquet(path)
    features = [c for c in emb.columns if c != "well_id"]
    emb = emb.groupby("well_id")[features].mean()
    df = meta.merge(emb, left_on="well_id", right_index=True)
    X = df[features].to_numpy(float)
    keep = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)  # CellProfiler has constant / NaN features
    return X[:, keep], df[meta.columns]


def recall(X: np.ndarray, obs: pd.DataFrame) -> pd.Series:
    crispr = (obs["perturbation_type"] == "CRISPR").to_numpy() & ~obs["perturbation"].str.contains("_control").to_numpy()
    genes = pd.DataFrame(X[crispr]).groupby(obs["perturbation"].to_numpy()[crispr]).mean()
    r = known_relationship_benchmark(
        Bunch(metadata=pd.DataFrame({"gene": genes.index}), features=genes.reset_index(drop=True)),
        pert_col="gene", recall_thr_pairs=[(0.05, 0.95)],
    )
    return r.set_index("source")["recall_0.05_0.95"]


def main():
    meta = pd.read_csv(CORE / "metadata_rxrx3_core.csv", low_memory=False)
    meta["perturbation"] = meta["treatment"].where(meta["treatment"].str.contains("_control"), meta["treatment"].str.split("_").str[0])
    meta["plate_batch"] = meta["experiment_name"] + "/" + meta["plate"].astype(str)
    sets = RELEASED | {Path(p).stem: p for p in sys.argv[1:]}
    rows = {}
    for name, file in sets.items():
        X, obs = wells(CORE / file, meta)
        print(f"{name}: {X.shape}", flush=True)
        aligned = {
            "PCA-CS experiment": pca_centerscale_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
            "PCA-CS plate": pca_centerscale_on_controls(X, obs, "perturbation", "EMPTY_control", "plate_batch"),
            "TVN experiment": tvn_on_controls(X, obs, "perturbation", "EMPTY_control", "experiment_name"),
            "none": X,
        }
        for how, Y in aligned.items():
            rows[(name, how)] = recall(Y, obs)
            print(name, how, rows[(name, how)].round(3).to_dict(), flush=True)
    table = pd.DataFrame(rows).T
    paper = PAPER.set_axis(pd.MultiIndex.from_product([PAPER.index, ["paper"]]))
    print(pd.concat([table, paper]).sort_index().round(3).to_string())
    table.to_csv(ROOT / "outputs/2026-11-15/rxrx3_core_repro.csv")


if __name__ == "__main__":
    main()
