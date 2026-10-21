import pandas as pd
import pytest
import torch

from genetic_perturbation_playground.data import rxrx1
from genetic_perturbation_playground.data.rxrx1 import crop, crop_obs


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


def test_load_rxrx1_missing_file_says_how_to_make_it(monkeypatch, tmp_path):
    monkeypatch.setattr(rxrx1, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="embed_rxrx1"):
        rxrx1.load_rxrx1(4)
