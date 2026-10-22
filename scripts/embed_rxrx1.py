"""Phenom embeddings of RxRx1, one row per image crop: the input to train_batch_transfer_rxrx1.

Each 512x512 6-channel site is cut into grid x grid crops in memory (crops are never written to
disk), each crop is embedded by the frozen OpenPhenom encoder at native resolution (model/phenom.py
rebuilds its positional table for the crop size), and only the embeddings are saved, to
data/rxrx1_phenom_<grid>x<grid>[_sitenorm][_nuclei<f>].h5ad (data/rxrx1.rxrx1_path) with per-crop
labels from data/rxrx1.crop_obs. No batch correction (OpenPhenom's README suggests
PCA-CenterScale/TVN): batch effects are what we model.

Each channel is standardised to mean 0, std 1 per crop, as OpenPhenom was trained, or with
standardise_sites per site before cropping, so a crop with few cells keeps its low intensity rather
than having its noise scaled up (self-standardisation, Sypetkowski et al. 2023). With
min_nuclei_fraction, crops whose nuclei cover less of them are dropped: the share of a crop above its
site's Otsu threshold in the DNA channel (Kim et al. 2025 drop crops under 1%).

Config: configs/embed_rxrx1.yaml. Usage:

    uv run python scripts/embed_rxrx1.py 'experiments=[HEPG2-01]'      # smoke run
    sbatch -J embed_rxrx1 --cpus-per-task=16 hpc/run.sbatch scripts/embed_rxrx1.py workers=16
"""

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from genetic_perturbation_playground.data.rxrx1 import (
    RXRX1_DIR,
    crop,
    crop_obs,
    load_sites,
    nuclear_fraction,
    rxrx1_path,
    site_prefixes,
    write_embeddings,
)
from genetic_perturbation_playground.model.phenom import embed, load_phenom, standardise


def embed_sites(model, x: torch.Tensor, cfg: DictConfig, device: str) -> tuple:
    """Embeddings of the crops of uint8 sites x (cpu) kept by cfg.min_nuclei_fraction, and the
    boolean mask of kept crops in crop() order."""
    keep = nuclear_fraction(x, cfg.grid) >= cfg.min_nuclei_fraction
    x = x.to(device)
    crops = (
        crop(standardise(x), cfg.grid) if cfg.standardise_sites else standardise(crop(x, cfg.grid))
    )
    return embed(model, crops[keep.to(device)]).cpu(), keep


@hydra.main(config_path="../configs", config_name="embed_rxrx1", version_base=None)
def main(cfg: DictConfig):
    meta = pd.read_csv(RXRX1_DIR / "metadata.csv")
    if cfg.experiments:
        meta = meta[meta["experiment"].isin(cfg.experiments)]
    prefixes = site_prefixes(meta)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_phenom(512 // cfg.grid).to(device)
    loader = torch.utils.data.DataLoader(
        prefixes, cfg.batch_size, num_workers=cfg.workers, collate_fn=load_sites, pin_memory=True
    )
    print(f"embedding {len(prefixes):,} sites x {cfg.grid**2} crops on {device}", flush=True)
    # disable=False overrides run.sbatch's TQDM_DISABLE; one refresh per 30s keeps the log short
    out = [
        embed_sites(model, x, cfg, device)
        for x in tqdm(loader, "batches", disable=False, mininterval=30)
    ]

    X, keep = (torch.cat(t).numpy() for t in zip(*out, strict=True))
    path = rxrx1_path(cfg.grid, cfg.standardise_sites, cfg.min_nuclei_fraction)
    write_embeddings(X, crop_obs(meta, cfg.grid)[keep], path)
    print(
        f"wrote {path}: {X.shape[0]:,} crops x {X.shape[1]} | {len(meta):,} sites"
        f" | dropped {(~keep).sum():,} crops under {cfg.min_nuclei_fraction} nuclei"
    )


if __name__ == "__main__":
    main()
