"""Whole-slide reading and tiling on top of TIAToolbox."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def open_slide(path: str | Path, mpp: float | None = None):
    """TIAToolbox reader for any supported format. `mpp` is used only when the slide does not record it."""
    from tiatoolbox.wsicore.wsireader import WSIReader

    reader = WSIReader.open(str(path))
    if reader.info.mpp is None or not np.all(np.isfinite(reader.info.mpp)):
        if mpp is None:
            raise ValueError(f"{Path(path).name}: slide has no microns-per-pixel; set `mpp` in the config")
        reader = WSIReader.open(str(path), mpp=(mpp, mpp))
    return reader


def slide_mpp(reader) -> float:
    return float(np.mean(reader.info.mpp))


def size_at(reader, mpp: float) -> tuple[int, int]:
    """(width, height) of the slide at `mpp`."""
    w, h = reader.slide_dimensions(resolution=mpp, units="mpp")
    return int(w), int(h)


def device_for(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── tiling with a mask ───────────────────────────────────────────────────────────────────────────────────────────────
def tile_grid(width: int, height: int, tile: int) -> np.ndarray:
    """[N, 4] tile bounds (x0, y0, x1, y1) covering the image at its own resolution, row-major."""
    xs = np.arange(0, width, tile)
    ys = np.arange(0, height, tile)
    gx, gy = np.meshgrid(xs, ys)
    x0, y0 = gx.ravel(), gy.ravel()
    return np.stack([x0, y0, x0 + tile, y0 + tile], 1).astype(np.int64)


def mask_fraction(mask: np.ndarray, bounds: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Fraction of each tile covered by mask pixels > 0. `mask` spans the whole slide; `bounds` are tiles in an image
    of `image_size` = (width, height). Parts of a tile beyond the slide edge count as background. Uses an integral
    image, so it is fast for any number of tiles."""
    m = (mask > 0).astype(np.float64)
    mh, mw = m.shape[:2]
    sx, sy = mw / image_size[0], mh / image_size[1]
    ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    x0, y0 = np.round(bounds[:, 0] * sx).astype(int), np.round(bounds[:, 1] * sy).astype(int)
    x1 = np.maximum(np.round(bounds[:, 2] * sx).astype(int), x0 + 1)
    y1 = np.maximum(np.round(bounds[:, 3] * sy).astype(int), y0 + 1)
    full = (x1 - x0) * (y1 - y0)
    cx0, cx1, cy0, cy1 = (np.clip(v, 0, n) for v, n in ((x0, mw), (x1, mw), (y0, mh), (y1, mh)))
    inside = ii[cy1, cx1] - ii[cy0, cx1] - ii[cy1, cx0] + ii[cy0, cx0]
    return inside / full


@dataclass
class TileSpec:
    tile: int
    mpp: float


class TileDataset(torch.utils.data.Dataset):
    """Reads tiles (bounds at `spec.mpp`) from one slide; each worker opens its own reader."""

    def __init__(self, path: str | Path, bounds: np.ndarray, spec: TileSpec, fallback_mpp: float | None = None):
        self.path, self.bounds, self.spec, self.fallback = str(path), bounds, spec, fallback_mpp
        self._reader = None

    def __len__(self):
        return len(self.bounds)

    def __getitem__(self, i):
        if self._reader is None:
            self._reader = open_slide(self.path, self.fallback)
        x0, y0, x1, y1 = (int(v) for v in self.bounds[i])
        img = self._reader.read_rect((x0, y0), (x1 - x0, y1 - y0), resolution=self.spec.mpp, units="mpp",
                                     coord_space="resolution")
        return torch.from_numpy(np.ascontiguousarray(img[..., :3])), i


def loader(ds: TileDataset, batch_size: int, workers: int, desc: str = ""):
    """Batches of tiles read by `workers` processes ahead of the network; a progress bar shows in terminals only."""
    import sys

    from tqdm import tqdm

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=False,
                                     persistent_workers=False, pin_memory=torch.cuda.is_available())
    return tqdm(dl, desc=desc, unit="batch", leave=False, disable=not sys.stderr.isatty())
