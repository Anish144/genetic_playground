"""OpenPhenom (Recursion's CA-MAE ViT-S/16, trained on 256px cell painting crops) as a frozen encoder.

Code and weights come from huggingface.co/recursionpharma/OpenPhenom into data/OpenPhenom/. Only the
encoder half of their MAEModel is built, so their transformers wrapper is not needed.
"""

import sys

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from genetic_perturbation_playground.data.data_loader import DATA_DIR

PHENOM_DIR = DATA_DIR / "OpenPhenom"


def load_phenom(crop_size: int) -> torch.nn.Module:
    """The OpenPhenom encoder (eval mode, cpu) for crop_size x crop_size inputs.

    The sin-cos positional table is rebuilt for the crop's patch grid: the model only slices its
    256px (16x16 patch) table, which would give the patches of a smaller crop wrong positions.
    Sin-cos is fixed, so positions 0..crop_size/16 - 1 are the ones it was trained on.
    """
    snapshot_download(
        "recursionpharma/OpenPhenom", local_dir=PHENOM_DIR, allow_patterns=["*.py", "*.safetensors"]
    )
    sys.path.insert(0, str(DATA_DIR))
    from OpenPhenom.mae_modules import MAEEncoder
    from OpenPhenom.vit import (
        generate_2d_sincos_pos_embeddings,
        sincos_positional_encoding_vit,
        vit_small_patch16_256,
    )

    model = MAEEncoder(
        sincos_positional_encoding_vit(vit_small_patch16_256(global_pool="avg")),
        max_in_chans=11,
        channel_agnostic=True,
    )
    # a torch pickle despite the name; a file handle stops torch dispatching on the extension
    with open(PHENOM_DIR / "model.safetensors", "rb") as f:
        state = torch.load(f, map_location="cpu")["state_dict"]
    model.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    )
    model.vit_backbone.pos_embed = generate_2d_sincos_pos_embeddings(
        model.embed_dim, length=crop_size // 16, use_class_token=True, num_modality=11
    )
    return model.eval()


@torch.no_grad()
def embed(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """(N, C, S, S) uint8 -> (N, 384), as MAEModel.predict: scale to [0, 1], standardise each crop's
    channels on their own, mean over patch tokens (class token dropped)."""
    x = F.instance_norm(x.float() / 255)
    return model.vit_backbone.forward_features(x)[:, 1:].mean(1)
