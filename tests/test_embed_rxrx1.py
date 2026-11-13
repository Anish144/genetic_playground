import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from genetic_perturbation_playground.data import rxrx1

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import embed_rxrx1 as mod  # noqa: E402


@pytest.mark.parametrize("standardise_sites", [False, True])
def test_embedding_keeps_blank_crops_and_labels_across_batches(
    tmp_path, monkeypatch, standardise_sites
):
    meta = pd.DataFrame(
        {
            "experiment": ["E1", "E2"],
            "plate": [1, 1],
            "well": ["B02", "B03"],
            "site": [1, 2],
            "cell_type": ["HEPG2"] * 2,
            "well_type": ["negative_control", "treatment"],
            "sirna": ["EMPTY", "s1"],
        }
    )
    meta.to_csv(tmp_path / "metadata.csv", index=False)
    monkeypatch.setattr(rxrx1, "RXRX1_DIR", tmp_path)
    monkeypatch.setattr(rxrx1, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "RXRX1_DIR", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "load_phenom", lambda size: torch.nn.Identity())

    # One wholly blank batch, then four crops with distinct bright-pixel positions.
    for i, prefix in enumerate(rxrx1.site_prefixes(meta)):
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((8, 8), dtype=np.uint8)
        if i == 1:
            pixels[[0, 0, 5, 5], [0, 5, 0, 5]] = 200
        for channel in range(1, 7):
            Image.fromarray(pixels).save(f"{prefix}_w{channel}.png")

    def embed_pixel_positions(model, crops):
        pixels = crops[:, 0].flatten(1)
        return torch.stack((pixels.argmax(1).float(), pixels.amax(1)), dim=1)

    monkeypatch.setattr(mod, "embed", embed_pixel_positions)
    cfg = OmegaConf.load(ROOT / "configs" / "embed_rxrx1.yaml")
    cfg.grid, cfg.batch_size, cfg.workers = 2, 1, 0
    cfg.standardise_sites = standardise_sites
    mod.main.__wrapped__(cfg)

    result = ad.read_h5ad(rxrx1.rxrx1_path(2, standardise_sites))
    assert result.shape == (8, 2)
    assert np.isfinite(result.X).all()
    np.testing.assert_array_equal(result.X[:4], 0)
    np.testing.assert_array_equal(result.X[4:, 0], [0, 1, 4, 5])
    assert (result.X[4:, 1] > 0).all()
    assert result.obs["crop"].tolist() == [0, 1, 2, 3] * 2
    assert result.obs["perturbation"].tolist() == ["control"] * 4 + ["s1"] * 4
    assert result.obs["batch"].tolist() == ["E1"] * 4 + ["E2"] * 4
    assert result.obs["site"].tolist() == [1] * 4 + [2] * 4
