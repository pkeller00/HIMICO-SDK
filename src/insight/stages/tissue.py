"""Stage 2: CRC tissue-type classification.

A ViT-B/16 classifies 224 px tiles at 0.5 µm/px (inside the stage-1 mask, >40% coverage) into eight colorectal tissue
types. Tiles whose most likely type is in `keep` (default TUM, STR, LYM, MUC) are painted into a tissue mask at
`mask_mpp`, which decides where stage-3 features are taken.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import weights
from ..wsi import TileDataset, TileSpec, loader, mask_fraction, open_slide, size_at, tile_grid

log = logging.getLogger("insight")
CLASSES = ["TUM", "NORM", "LYM", "MUS", "STR", "DEB", "MUC", "ADI"]


class CRCTissue:
    def __init__(self, device: torch.device, weights_dir: str | None = None):
        import timm
        from safetensors.torch import load_file

        self.device = device
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASSES),
                                       norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-12))
        self.model.load_state_dict(load_file(weights.resolve("crc_tissue/vit_b16.safetensors", weights_dir)))
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - 0.5) / 0.5                     # ViT-in21k image processor: mean = std = 0.5
        return self.model(x).softmax(-1).float().cpu()


def run_tissue(model: CRCTissue, slide, mask_path: Path, out: Path, cfg, batch_size: int, workers: int,
               fallback_mpp: float | None) -> dict:
    reader = open_slide(slide.path, fallback_mpp)
    W, H = size_at(reader, cfg.mpp)
    grid = tile_grid(W, H, cfg.tile)
    mask = np.array(Image.open(mask_path).convert("L"))
    sel = grid[mask_fraction(mask, grid, (W, H)) > cfg.min_mask_fraction]
    probs = np.zeros((len(sel), len(CLASSES)), np.float32)
    if len(sel):
        ds = TileDataset(slide.path, sel, TileSpec(cfg.tile, cfg.mpp), fallback_mpp)
        for imgs, idx in loader(ds, batch_size, workers, desc=f"tissue {slide.id}"):
            probs[idx.numpy()] = model(imgs).numpy()
    np.savez_compressed(out / f"{slide.id}.npz", bounds=sel, probs=probs.astype(np.float16), classes=np.array(CLASSES),
                        mpp=cfg.mpp)
    # tissue mask of the selected types at mask_mpp
    mw, mh = size_at(reader, cfg.mask_mpp)
    tmask = np.zeros((mh, mw), np.uint8)
    keep = np.isin(np.array(CLASSES)[probs.argmax(1)], cfg.keep) if len(sel) else np.zeros(0, bool)
    s = cfg.mpp / cfg.mask_mpp
    for x0, y0, x1, y1 in sel[keep]:
        tmask[int(y0 * s):int(np.ceil(y1 * s)), int(x0 * s):int(np.ceil(x1 * s))] = 255
    Image.fromarray(tmask).save(out / f"{slide.id}_mask.png")
    if cfg.write_overlay and len(sel):
        _write_overlay(out / f"{slide.id}.db", sel, probs, cfg.mpp / float(np.mean(reader.info.mpp)))
    counts = {c: int(n) for c, n in zip(CLASSES, np.bincount(probs.argmax(1), minlength=len(CLASSES)))} if len(sel) else {}
    return {"tiles": int(len(sel)), "kept_tiles": int(keep.sum()), **{f"n_{k}": v for k, v in counts.items()}}


def _write_overlay(path: Path, bounds: np.ndarray, probs: np.ndarray, to_baseline: float):
    """TIAToolbox annotation store: one polygon per tile with class probabilities and `max_class`."""
    from shapely.geometry import box
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    store = SQLiteStore()
    anns = []
    for (x0, y0, x1, y1), p in zip(bounds * to_baseline, probs):
        props = {c: float(v) for c, v in zip(CLASSES, p)}
        props["max_class"] = CLASSES[int(np.argmax(p))]
        anns.append(Annotation(box(x0, y0, x1, y1), props))
    store.append_many(anns)
    store.dump(str(path))
    store.close()
