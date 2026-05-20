from pathlib import Path

import scanpy as sc

from .perturbation_datasets import replogle_load_and_preprocess

_DATASETS: dict[str, tuple[callable, str]] = {
    "replogle_k562": (replogle_load_and_preprocess, "replogle_k562_preprocessed.h5ad"),
}


def _data_dir() -> Path:
    _pkg_root = Path(__file__).parents[3]  # src/genetic_perturbation_playground/data/ -> project root
    candidates = [Path("data"), Path("../data"), _pkg_root / "data"]
    data_dir = next((p for p in candidates if p.is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError(f"Cannot locate data/ directory from CWD: {Path.cwd()}")
    return data_dir


def _try_load(path: Path) -> sc.AnnData | None:
    try:
        adata = sc.read_h5ad(path)
        assert "perturbation" in adata.obs.columns
        return adata
    except Exception as e:
        print(f"Cached file invalid ({e}); deleting and reprocessing…")
        path.unlink(missing_ok=True)
        return None


def load_dataset(name: str = "replogle_k562", **preprocess_kwargs) -> sc.AnnData:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(_DATASETS)}")

    preprocess_fn, cache_filename = _DATASETS[name]
    cache_path = _data_dir() / cache_filename

    adata = _try_load(cache_path) if cache_path.exists() else None
    if adata is None:
        adata = preprocess_fn(**preprocess_kwargs)
        adata.write_h5ad(cache_path, compression="gzip")
        print(f"Saved to {cache_path}")
    else:
        print(f"Loaded from {cache_path}")

    print(f"  {adata.n_obs:,} cells  ×  {adata.n_vars:,} genes  |  {adata.obs['perturbation'].nunique():,} perturbations")
    return adata
