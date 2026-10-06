from pathlib import Path

import scanpy as sc

from .perturbation_datasets import replogle_load_and_preprocess

CACHE_PATH = Path(__file__).parents[3] / "data" / "replogle_k562_preprocessed.h5ad"


def _try_load(path: Path) -> sc.AnnData | None:
    try:
        adata = sc.read_h5ad(path)
        assert "perturbation" in adata.obs.columns
        return adata
    except Exception as e:
        print(f"Cached file invalid ({e}); deleting and reprocessing…")
        path.unlink(missing_ok=True)
        return None


def load_dataset() -> sc.AnnData:
    """Replogle 2022 K562 essential screen, preprocessed once and cached under data/."""
    adata = _try_load(CACHE_PATH) if CACHE_PATH.exists() else None
    if adata is None:
        adata = replogle_load_and_preprocess()
        CACHE_PATH.parent.mkdir(exist_ok=True)
        adata.write_h5ad(CACHE_PATH, compression="gzip")
        print(f"Saved to {CACHE_PATH}")
    else:
        print(f"Loaded from {CACHE_PATH}")

    print(f"  {adata.n_obs:,} cells  ×  {adata.n_vars:,} genes  |  {adata.obs['perturbation'].nunique():,} perturbations")
    return adata
