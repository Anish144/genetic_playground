"""Downloads OpenPhenom (~700MB, encoder + decoder) into data/OpenPhenom on first run."""

import torch

from genetic_perturbation_playground.model.phenom import embed, load_phenom


def test_positional_table_matches_crop_size():
    model = load_phenom(128)
    assert model.vit_backbone.pos_embed.shape == (1, 1 + 11 * 8 * 8, 384)


def test_embed_shape_and_finite():
    model = load_phenom(128)
    x = torch.randint(0, 255, (3, 6, 128, 128), dtype=torch.uint8)
    out = embed(model, x)
    assert out.shape == (3, 384) and torch.isfinite(out).all()
