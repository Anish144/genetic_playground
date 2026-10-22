"""Phenom embeddings of RxRx1, one row per image crop: the input to train_batch_transfer_rxrx1.

Each 512x512 6-channel site is cut into grid x grid crops in memory (crops are never written to
disk), each crop is embedded by the frozen OpenPhenom encoder at native resolution (model/phenom.py
rebuilds its positional table for the crop size), and only the embeddings are saved, to
data/rxrx1_phenom_<grid>x<grid>.h5ad with per-crop labels from data/rxrx1.crop_obs. No batch
correction (OpenPhenom's README suggests PCA-CenterScale/TVN): batch effects are what we model.

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
    rxrx1_path,
    site_prefixes,
    write_embeddings,
)
from genetic_perturbation_playground.model.phenom import embed, load_phenom


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
    X = [
        embed(model, crop(x.to(device), cfg.grid)).cpu()
        for x in tqdm(loader, "batches", disable=False, mininterval=30)
    ]

    X, path = torch.cat(X).numpy(), rxrx1_path(cfg.grid)
    write_embeddings(X, crop_obs(meta, cfg.grid), path)
    print(f"wrote {path}: {X.shape[0]:,} crops x {X.shape[1]} | {len(meta):,} sites")


if __name__ == "__main__":
    main()
