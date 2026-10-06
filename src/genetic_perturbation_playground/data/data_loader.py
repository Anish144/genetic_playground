from pathlib import Path

import scanpy as sc

from .perturbation_datasets import replogle_load_and_preprocess

CACHE_PATH = Path(__file__).parents[3] / "data" / "replogle_k562_preprocessed.h5ad"


def load_dataset() -> sc.AnnData:
    """Replogle 2022 K562 essential screen, preprocessed once and cached under data/."""
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(exist_ok=True)
        replogle_load_and_preprocess().write_h5ad(CACHE_PATH, compression="gzip")
    adata = sc.read_h5ad(CACHE_PATH)
    print(f"Loaded {CACHE_PATH}")
    print(f"  {adata.n_obs:,} cells  ×  {adata.n_vars:,} genes  |  {adata.obs['perturbation'].nunique():,} perturbations")
    return adata
