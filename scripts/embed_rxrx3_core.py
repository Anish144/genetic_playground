"""OpenPhenom embeddings of the public RxRx3-core images, as arXiv 2503.20158 describes: each
512x512x6 image split into four 256x256 crops, crop embeddings (model.predict: mean of patch
tokens) averaged per image, then per well. Writes rxrx3_core/ours_openphenom.parquet (well_id +
feature_i), the same layout as the released embeddings.

Image shards: rxrx3_core/data/train-*.parquet (the Hugging Face layout), one row per channel image, __key__
"<experiment>/Plate<p>/<address>_s<site>_<channel>", jp2 = {"bytes": JPEG 2000}.

    uv run python scripts/embed_rxrx3_core.py    (one GPU; bf16)
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from genetic_perturbation_playground.model.phenom import embed, load_phenom, standardise

CORE = Path(__file__).parents[1] / "outputs/scratch/rxrx3_core"


def crop(x: torch.Tensor, grid: int) -> torch.Tensor:
    """(N, C, H, W) -> (N * grid**2, C, H / grid, W / grid); each image's crops row-major."""
    n, c, h, w = x.shape
    x = x.reshape(n, c, grid, h // grid, grid, w // grid).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(-1, c, h // grid, w // grid)


def images(shard: Path) -> tuple[list[str], np.ndarray]:
    """The shard's sites as (well_id per site, (N, 6, 512, 512) uint8)."""
    df = pd.read_parquet(shard)
    key = df["__key__"].str.extract(r"(?P<exp>[^/]+)/Plate(?P<plate>\d+)/(?P<well>[^_]+)_s(?P<site>\d+)_(?P<ch>\d+)")
    df = pd.concat([df, key], axis=1).sort_values(["exp", "plate", "well", "site", "ch"])
    df["site_id"] = df["exp"] + "_" + df["plate"] + "_" + df["well"] + "_" + df["site"]
    sites = df.groupby("site_id", sort=False)
    complete = sites["ch"].transform("size") == 6
    print(f"{shard.name}: dropping {(~complete).sum()} channel images of sites split across shards", flush=True)
    df = df[complete]
    pixels = np.stack([np.array(Image.open(io.BytesIO(b["bytes"]))) for b in df["jp2"]])
    pixels = pixels.reshape(-1, 6, *pixels.shape[1:])
    well_ids = (df["exp"] + "_" + df["plate"] + "_" + df["well"]).to_numpy()[::6]
    return list(well_ids), pixels


def main():
    device = "cuda"
    model = load_phenom(256).to(device)
    shards = sorted((CORE / "data").glob("train-*.parquet"))
    out = []
    for i, shard in enumerate(shards):
        well_ids, pixels = images(shard)
        feats = []
        for batch in np.array_split(np.arange(len(pixels)), max(1, len(pixels) // 64)):
            x = crop(torch.from_numpy(pixels[batch]).to(device), 2)  # (B*4, 6, 256, 256)
            with torch.autocast(device, dtype=torch.bfloat16):
                z = embed(model, standardise(x)).float()
            feats.append(z.view(len(batch), 4, -1).mean(1).cpu().numpy())
        out.append(pd.DataFrame(np.concatenate(feats)).assign(well_id=well_ids))
        print(f"{i + 1}/{len(shards)} {shard.name}: {len(well_ids)} sites", flush=True)
    df = pd.concat(out).groupby("well_id").mean()
    df.columns = [f"feature_{c}" for c in df.columns]
    df.reset_index().to_parquet(CORE / "ours_openphenom.parquet")
    print(f"{len(df)} wells", file=sys.stderr)


if __name__ == "__main__":
    main()
