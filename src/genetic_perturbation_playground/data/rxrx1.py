"""RxRx1 (Recursion, 6-channel cell painting, 1,138 siRNAs x 51 experiments) and its crop embeddings.

Images: images/<experiment>/Plate<plate>/<well>_s<site>_w<1..6>.png, 512x512 uint8, one png per
channel. Embeddings are written by scripts/embed_rxrx1.py, one row per crop.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PIL import Image

from genetic_perturbation_playground.data.data_loader import DATA_DIR

RXRX1_DIR = DATA_DIR / "rxrx1" / "rxrx1"


def rxrx1_path(grid: int) -> Path:
    """Embeddings of every site cut into grid x grid crops."""
    return DATA_DIR / f"rxrx1_phenom_{grid}x{grid}.h5ad"


def site_prefixes(meta: pd.DataFrame) -> list[str]:
    """Path of each metadata row's site without the channel suffix, e.g. .../Plate1/B02_s1."""
    return [
        str(RXRX1_DIR / "images" / e / f"Plate{p}" / f"{w}_s{s}")
        for e, p, w, s in meta[["experiment", "plate", "well", "site"]].itertuples(index=False)
    ]


def load_sites(prefixes: list[str]) -> torch.Tensor:
    """(sites, 6, 512, 512) uint8 from the w1..w6 pngs. Used as DataLoader collate_fn (in workers)."""
    return torch.from_numpy(
        np.stack([[np.asarray(Image.open(f"{p}_w{c}.png")) for c in range(1, 7)] for p in prefixes])
    )


def crop(x: torch.Tensor, grid: int) -> torch.Tensor:
    """(N, C, H, W) -> (N * grid**2, C, H / grid, W / grid); each image's crops row-major."""
    n, c, h, w = x.shape
    x = x.reshape(n, c, grid, h // grid, grid, w // grid).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(-1, c, h // grid, w // grid)


def crop_obs(meta: pd.DataFrame, grid: int) -> pd.DataFrame:
    """One row per crop, in crop() order: the site's metadata repeated grid**2 times, plus crop.

    perturbation is the siRNA, or "control" for negative-control (EMPTY) wells; positive controls
    stay perturbations. batch is the experiment.
    """
    n_crops = grid**2
    sites = meta.iloc[np.repeat(np.arange(len(meta)), n_crops)].reset_index(drop=True)
    is_ctrl = sites["well_type"] == "negative_control"
    obs = pd.DataFrame(
        {
            "perturbation": np.where(is_ctrl, "control", sites["sirna"]),
            "batch": sites["experiment"],
            **sites[["cell_type", "well_type", "plate", "well", "site"]],
            "crop": np.tile(np.arange(n_crops), len(meta)),
        }
    )
    # pandas 3 strings are arrow-backed, which anndata 0.12 can't write: index and categories as object
    obs.index = obs.index.astype(str).astype(object)
    for col in ["perturbation", "batch", "cell_type", "well_type", "well"]:
        obs[col] = pd.Categorical(obs[col], pd.Index(sorted(set(obs[col])), dtype=object))
    return obs


def write_embeddings(X: np.ndarray, obs: pd.DataFrame, path: Path) -> None:
    """Crop embeddings X (crops x features) with their crop_obs labels, as h5ad."""
    var = pd.DataFrame(index=pd.Index([f"phenom_{i}" for i in range(X.shape[1])], dtype=object))
    ad.AnnData(X, obs=obs, var=var).write_h5ad(path)


def load_rxrx1(grid: int = 4) -> sc.AnnData:
    """RxRx1 Phenom embeddings, one row per crop, written by scripts/embed_rxrx1.py."""
    path = rxrx1_path(grid)
    adata = sc.read_h5ad(path)
    print(f"Loaded {path}")
    print(
        f"  {adata.n_obs:,} crops  ×  {adata.n_vars:,} features  |  {adata.obs['perturbation'].nunique():,} perturbations"
    )
    return adata
