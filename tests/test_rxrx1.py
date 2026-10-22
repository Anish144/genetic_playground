import anndata as ad
import numpy as np
import pandas as pd
import torch

from genetic_perturbation_playground.data.data_loader import DATA_DIR
from genetic_perturbation_playground.data.rxrx1 import (
    crop,
    crop_obs,
    nuclear_fraction,
    rxrx1_path,
    write_embeddings,
)


def test_crop_matches_slicing():
    x = torch.arange(2 * 6 * 8 * 8).reshape(2, 6, 8, 8)
    out = crop(x, 2)
    assert out.shape == (8, 6, 4, 4)
    assert torch.equal(out[0], x[0, :, :4, :4])
    assert torch.equal(out[1], x[0, :, :4, 4:])  # row-major: second crop is top-right
    assert torch.equal(out[2], x[0, :, 4:, :4])
    assert torch.equal(out[7], x[1, :, 4:, 4:])


def test_crop_obs_labels():
    meta = pd.DataFrame(
        {
            "experiment": ["E1", "E1"],
            "plate": [1, 1],
            "well": ["B02", "B03"],
            "site": [1, 1],
            "cell_type": ["HEPG2"] * 2,
            "well_type": ["negative_control", "treatment"],
            "sirna": ["EMPTY", "s1"],
        }
    )
    obs = crop_obs(meta, 2)
    assert len(obs) == 8 and obs.index.is_unique
    assert list(obs["perturbation"].astype(str)) == ["control"] * 4 + ["s1"] * 4
    assert list(obs["crop"]) == [0, 1, 2, 3] * 2
    assert (obs["batch"].astype(str) == "E1").all()


def test_write_embeddings_round_trips(tmp_path):
    meta = pd.DataFrame(
        {"experiment": ["E1"], "plate": [1], "well": ["B02"], "site": [1], "cell_type": ["HEPG2"]}
        | {"well_type": ["treatment"], "sirna": ["s1"]}
    )
    obs = crop_obs(meta, 2)
    write_embeddings(np.zeros((len(obs), 3), dtype=np.float32), obs, tmp_path / "x.h5ad")
    back = ad.read_h5ad(tmp_path / "x.h5ad")
    assert list(back.obs["crop"]) == [0, 1, 2, 3]
    assert list(back.obs["perturbation"].astype(str)) == ["s1"] * 4


def test_nuclear_fraction_is_the_share_of_each_crop_above_its_sites_threshold():
    x = torch.full((1, 6, 8, 8), 10, dtype=torch.uint8)
    x[0, 0, :4, :4] = 200  # nuclei fill the top-left crop of the DNA channel
    x[0, 1:, 4:, 4:] = 250  # other channels never count
    assert nuclear_fraction(x, 2).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_rxrx1_path_names_each_embedding_variant():
    assert rxrx1_path(4) == DATA_DIR / "rxrx1_phenom_4x4.h5ad"
    assert rxrx1_path(4, True, 0.01).name == "rxrx1_phenom_4x4_sitenorm_nuclei0.01.h5ad"
