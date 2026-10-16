"""Prediction outputs: per-slide TIAToolbox overlays and heatmap images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def rdbu(v: np.ndarray) -> np.ndarray:
    """Fixed 0-1 colour scale, blue (0, lower) to red (1, higher), as on the INSIGHT website."""
    from matplotlib import colormaps

    return (colormaps["RdBu_r"](np.clip(v, 0, 1))[..., :3] * 255).astype(np.uint8)


def write_overlay(path: Path, bounds_baseline: np.ndarray, values: np.ndarray, name: str = "insight_risk"):
    """One square per patch with the local value as the single property `name`; open in the TIAToolbox viewer."""
    from shapely.geometry import box
    from tiatoolbox.annotation.storage import Annotation, SQLiteStore

    store = SQLiteStore()
    store.append_many([Annotation(box(*b), {name: float(v)}) for b, v in zip(bounds_baseline, values)])
    store.dump(str(path))
    store.close()


def write_heatmap(path: Path, bounds: np.ndarray, values: np.ndarray, max_side: int = 2000):
    """PNG heatmap (transparent background) from patch bounds at the feature resolution."""
    if not len(bounds):
        return
    x0, y0 = bounds[:, 0].min(), bounds[:, 1].min()
    x1, y1 = bounds[:, 2].max(), bounds[:, 3].max()
    s = max_side / max(x1 - x0, y1 - y0)
    w, h = max(1, int(np.ceil((x1 - x0) * s))), max(1, int(np.ceil((y1 - y0) * s)))
    img = np.zeros((h, w, 4), np.uint8)
    cols = rdbu(values)
    for (a, b, c, d), col in zip(bounds, cols):
        img[int((b - y0) * s):int(np.ceil((d - y0) * s)), int((a - x0) * s):int(np.ceil((c - x0) * s))] = (*col, 255)
    Image.fromarray(img, "RGBA").save(path)


def write_thumbnail(path: Path, reader, bounds_baseline: np.ndarray, values: np.ndarray, max_side: int = 1600,
                    alpha: float = 0.55):
    """Slide thumbnail with the per-patch values blended on top (fixed 0-1 colour scale)."""
    W0, H0 = (int(v) for v in reader.info.slide_dimensions)
    mpp = float(np.mean(reader.info.mpp)) * max(W0, H0) / max_side
    thumb = reader.slide_thumbnail(resolution=mpp, units="mpp")[..., :3].astype(np.float32)
    sy, sx = thumb.shape[0] / H0, thumb.shape[1] / W0
    over = thumb.copy()
    hit = np.zeros(thumb.shape[:2], bool)
    for (x0, y0, x1, y1), col in zip(bounds_baseline, rdbu(values)):
        ys, xs = slice(int(y0 * sy), int(np.ceil(y1 * sy))), slice(int(x0 * sx), int(np.ceil(x1 * sx)))
        over[ys, xs] = col
        hit[ys, xs] = True
    out = np.where(hit[..., None], (1 - alpha) * thumb + alpha * over, thumb)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(path, quality=88)
