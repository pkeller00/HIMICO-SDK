"""Stage 1: slide-level mask of usable tissue.

GrandQC (Weng et al., Nat Commun 2024; CC BY-NC-SA 4.0) in two steps, as in the INSIGHT pipeline:
  1. tissue detection on a JPEG-compressed thumbnail at 10 µm/px (U-Net++, 2 classes: 0 tissue, 1 background)
  2. artefact segmentation at `qc_mpp` (U-Net, classes 1 normal, 2 fold, 3 dark spot/foreign object, 4 pen marking,
     5 edge/air bubble, 6 out of focus, 7 background), evaluated only where step 1 found tissue
The final mask keeps `keep_classes` (default 1, 2, 5, 6) and is saved as a binary PNG at `qc_mpp`.

Alternatively masks can come from a folder: <slide_id>.png covering the whole slide at any resolution.
"""
from __future__ import annotations

import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import weights
from ..wsi import open_slide, slide_mpp

log = logging.getLogger("himico")
Image.MAX_IMAGE_PIXELS = None

TD_MPP = 10.0
TILE = 512
BACKGROUND = 7
QC_COLOURS = np.array([[0, 0, 0], [128, 128, 128], [255, 99, 71], [0, 255, 0], [255, 0, 0], [255, 0, 255], [75, 0, 130],
                       [255, 255, 255]], np.uint8)   # index = class


class GrandQC:
    def __init__(self, device: torch.device, qc_mpp: float = 1.5, weights_dir: str | None = None):
        import segmentation_models_pytorch as smp
        from safetensors.torch import load_file

        if qc_mpp not in (1.0, 1.5, 2.0):
            raise ValueError("mask.qc_mpp must be 1.0, 1.5 or 2.0")
        self.device, self.qc_mpp = device, qc_mpp
        self.td = smp.UnetPlusPlus(encoder_name="timm-efficientnet-b0", encoder_weights=None, classes=2)
        self.td.load_state_dict(load_file(weights.resolve("grandqc/td_mpp10.safetensors", weights_dir)))
        self.qc = smp.Unet(encoder_name="timm-efficientnet-b0", encoder_weights=None, classes=8)
        key = {1.0: "qc_mpp1", 1.5: "qc_mpp1.5", 2.0: "qc_mpp2"}[qc_mpp]
        self.qc.load_state_dict(load_file(weights.resolve(f"grandqc/{key}.safetensors", weights_dir)))
        self.td.to(device).eval()
        self.qc.to(device).eval()
        p = smp.encoders.get_preprocessing_params("timm-efficientnet-b0", "imagenet")
        self.mean = torch.tensor(p["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(p["std"], device=device).view(1, 3, 1, 1)

    def _prep(self, tiles: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(tiles).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _segment(self, model, tiles: np.ndarray, batch: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(tiles), batch):
            out.append(model(self._prep(tiles[i:i + batch])).argmax(1).cpu().numpy().astype(np.uint8))
        return np.concatenate(out) if out else np.zeros((0, TILE, TILE), np.uint8)

    def tissue(self, reader) -> tuple[np.ndarray, np.ndarray]:
        """Tissue map at 10 µm/px (0 tissue, 1 background) and the thumbnail it was computed on."""
        thumb = reader.slide_thumbnail(resolution=TD_MPP, units="mpp")[..., :3]
        # the detector was trained on JPEG-compressed thumbnails, so reproduce that step
        buf = io.BytesIO()
        Image.fromarray(thumb).save(buf, "JPEG", quality=80)
        img = np.array(Image.open(buf).convert("RGB"))
        h, w = img.shape[:2]
        pad = np.full((max(h, TILE), max(w, TILE), 3), 255, np.uint8)
        pad[:h, :w] = img
        H, W = pad.shape[:2]
        # tiles on a 512 grid; the last row/column is aligned to the image edge (overlapping), as in GrandQC
        xs = sorted({*range(0, W - TILE + 1, TILE), W - TILE})
        ys = sorted({*range(0, H - TILE + 1, TILE), H - TILE})
        coords = [(x, y) for y in ys for x in xs]
        pred = self._segment(self.td, np.stack([pad[y:y + TILE, x:x + TILE] for x, y in coords]))
        out = np.ones((H, W), np.uint8)
        for (x, y), p in zip(coords, pred):
            out[y:y + TILE, x:x + TILE] = p
        return out[:h, :w], thumb

    def _qc_tile(self, reader, i: int, j: int, p_s: int) -> np.ndarray:
        """As GrandQC: read the tile at full resolution, then LANCZOS down to 512 px. (Letting the reader resample
        instead makes tiles softer and inflates the out-of-focus class.)"""
        full = reader.read_rect((i * p_s, j * p_s), (p_s, p_s), resolution=0, units="level")[..., :3]
        return np.array(Image.fromarray(full).resize((TILE, TILE), Image.Resampling.LANCZOS))

    def __call__(self, reader, path: str | None = None, fallback_mpp: float | None = None, workers: int = 4,
                 batch: int = 16) -> dict[str, np.ndarray]:
        """`workers` threads read and LANCZOS-resize tiles ahead of the network (each thread has its own reader when
        `path` is given); results are identical to processing one tile at a time."""
        td, thumb = self.tissue(reader)
        mpp = float(np.mean(reader.info.mpp))
        p_s = int(self.qc_mpp / mpp * TILE)              # tile size at full resolution
        W0, H0 = (int(v) for v in reader.info.slide_dimensions)
        nx, ny = W0 // p_s, H0 // p_s
        W, H = int(W0 * mpp / self.qc_mpp), int(H0 * mpp / self.qc_mpp)
        td_q = np.array(Image.fromarray(td).resize((W, H), Image.Resampling.NEAREST))
        classes = np.zeros((H, W), np.uint8)            # beyond the last full tile: 0 (dropped), as in GrandQC
        coords = []
        for j in range(ny):
            for i in range(nx):
                t = td_q[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
                if np.count_nonzero(t == 0) > 50:
                    coords.append((i, j))
                else:
                    classes[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = BACKGROUND

        local = threading.local()

        def read(ij):
            r = reader
            if path is not None and workers > 1:
                r = getattr(local, "reader", None)
                if r is None:
                    r = local.reader = open_slide(path, fallback_mpp)
            return self._qc_tile(r, ij[0], ij[1], p_s)

        def write(chunk, tiles):
            for (i, j), p in zip(chunk, self._segment(self.qc, np.stack(tiles), batch=batch)):
                t = td_q[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
                classes[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = np.where(t == 1, BACKGROUND, p)

        if workers > 1 and path is not None:
            with ThreadPoolExecutor(workers) as pool:
                it = pool.map(read, coords)                  # ordered, read ahead while the network runs
                for k in range(0, len(coords), batch):
                    chunk = coords[k:k + batch]
                    write(chunk, [next(it) for _ in chunk])
        else:
            for k in range(0, len(coords), batch):
                chunk = coords[k:k + batch]
                write(chunk, [read(ij) for ij in chunk])
        return {"classes": classes, "tissue": td, "thumbnail": thumb}


def qc_overlay(thumb: np.ndarray, classes: np.ndarray) -> Image.Image:
    col = Image.fromarray(QC_COLOURS[np.clip(classes, 0, 7)]).resize(thumb.shape[1::-1], Image.Resampling.NEAREST)
    return Image.blend(Image.fromarray(thumb), col, 0.3)


def run_grandqc(model: GrandQC, slide, out: Path, keep: list[int], fallback_mpp: float | None, workers: int = 4,
                batch: int = 16):
    """Write <out>/<slide>.png (binary mask) plus QC class map, overlay and per-class areas."""
    reader = open_slide(slide.path, fallback_mpp)
    res = model(reader, str(slide.path), fallback_mpp, workers=max(1, workers), batch=batch)
    mask = np.isin(res["classes"], keep).astype(np.uint8) * 255
    qc_dir = out / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out / f"{slide.id}.png")
    Image.fromarray(res["classes"]).save(qc_dir / f"{slide.id}_classes.png")
    qc_overlay(res["thumbnail"], res["classes"]).save(qc_dir / f"{slide.id}_overlay.jpg", quality=85)
    px_mm2 = (model.qc_mpp / 1000) ** 2
    counts = np.bincount(res["classes"].ravel(), minlength=8)
    names = ["padding", "normal", "fold", "dark_spot", "pen", "edge_bubble", "out_of_focus", "background"]
    return {"slide_mpp": slide_mpp(reader), "kept_mm2": round(float(mask.astype(bool).sum() * px_mm2), 3),
            **{f"{n}_mm2": round(float(c * px_mm2), 3) for n, c in zip(names[1:7], counts[1:7])}}


def use_folder_mask(src_dir: Path, slide, out: Path) -> dict:
    """Copy a user-provided mask (<slide_id>.png in `src_dir`) into the workspace as a binary PNG."""
    cands = [src_dir / f"{slide.id}.png", src_dir / f"{Path(slide.path).name}.png", src_dir / f"{Path(slide.path).stem}.png"]
    src = next((c for c in cands if c.exists()), None)
    if src is None:
        raise FileNotFoundError(f"no mask for {slide.id} in {src_dir} (expected {slide.id}.png)")
    m = np.array(Image.open(src).convert("L"))
    Image.fromarray(((m > 0) * 255).astype(np.uint8)).save(out / f"{slide.id}.png")
    return {"source": str(src)}
