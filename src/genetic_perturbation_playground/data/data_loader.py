from pathlib import Path

import pertpy as pt
import scanpy as sc

CACHE_PATH = Path(__file__).parents[3] / "data" / "replogle_k562_preprocessed.h5ad"


def replogle_load_and_preprocess() -> sc.AnnData:
    adata = pt.data.replogle_2022_k562_essential()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=4000.0)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
    return adata[(adata.obs["nperts"] == 1) | (adata.obs["perturbation"] == "control")].copy()


def load_dataset() -> sc.AnnData:
    """Replogle 2022 K562 essential screen, preprocessed once and cached under data/."""
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(exist_ok=True)
        replogle_load_and_preprocess().write_h5ad(CACHE_PATH, compression="gzip")
    adata = sc.read_h5ad(CACHE_PATH)
    print(f"Loaded {CACHE_PATH}")
    print(f"  {adata.n_obs:,} cells  ×  {adata.n_vars:,} genes  |  {adata.obs['perturbation'].nunique():,} perturbations")
    return adata
