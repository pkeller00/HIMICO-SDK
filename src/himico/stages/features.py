"""Stage 3: patch embeddings with a registered feature extractor (default UNI)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import weights
from ..registry import register_features
from ..wsi import TileDataset, TileSpec, loader, mask_fraction, open_slide, size_at, tile_grid

log = logging.getLogger("himico")


class FeatureExtractor:
    """Base class: subclasses set `dim` and implement `__call__(uint8 NHWC batch) -> float [N, dim]`."""

    name = "base"
    dim = 0

    def __init__(self, device: torch.device, weights_dir: str | None = None):
        self.device = device


@register_features("uni")
class UNI(FeatureExtractor):
    """UNI ViT-L/16 (Chen et al., Nat Med 2024). Weights are gated by MahmoodLab: request access on Hugging Face and
    log in, or place pytorch_model.bin at <weights_dir>/uni/pytorch_model.bin."""

    dim = 1024

    def __init__(self, device: torch.device, weights_dir: str | None = None):
        super().__init__(device, weights_dir)
        import timm

        self.model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
                                       num_classes=0, dynamic_img_size=True)
        sd = torch.load(weights.resolve("uni/pytorch_model.bin", weights_dir), map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        return self.model((x - self.mean) / self.std).float().cpu()


def run_features(extractor: FeatureExtractor, slide, tissue_mask: Path, out: Path, cfg, batch_size: int, workers: int,
                 fallback_mpp: float | None) -> dict:
    reader = open_slide(slide.path, fallback_mpp)
    W, H = size_at(reader, cfg.mpp)
    grid = tile_grid(W, H, cfg.tile)
    mask = np.array(Image.open(tissue_mask).convert("L"))
    sel = grid[mask_fraction(mask, grid, (W, H)) > cfg.min_mask_fraction]
    feats = np.zeros((len(sel), extractor.dim), np.float32)
    if len(sel):
        ds = TileDataset(slide.path, sel, TileSpec(cfg.tile, cfg.mpp), fallback_mpp)
        for imgs, idx in loader(ds, batch_size, workers, desc=f"features {slide.id}"):
            feats[idx.numpy()] = extractor(imgs).numpy()
    np.savez(out / f"{slide.id}.npz", bounds=sel, features=feats, mpp=cfg.mpp,
             baseline_scale=cfg.mpp / float(np.mean(reader.info.mpp)))
    return {"patches": int(len(sel))}
