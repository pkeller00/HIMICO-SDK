"""Virtual staining of a whole slide with an image-to-image model (e.g. HistoChrome, H&E -> multiplex IHC).

As in HistoChromeCode's infer.py: tiles of `model.tile` px at `model.mpp` with `model.overlap` px overlap, kept when
their Otsu tissue fraction is at least `model.min_tissue` (TIAToolbox SlidingWindowPatchExtractor, same grid), model
outputs clamped to [0, 1], overlaps blended with a Hann window, background white, saved as a pyramidal OME-TIFF
(JPEG, quality 87) at the model resolution.

Unlike the original, tiles are stitched in a rolling band of tile rows instead of a full-slide float canvas, so the
only large file is the uint8 output itself (memory ~ slide width x tile x 16 bytes, e.g. ~0.7 GB for a 50k px wide
slide at 0.5 um/px).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..wsi import TileDataset, TileSpec, loader, open_slide, size_at

log = logging.getLogger("himico")


def tile_bounds(reader, model) -> np.ndarray:
    """[N, 4] tile bounds at the model resolution, row-major, tissue tiles only (same grid as the original)."""
    from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor

    ex = SlidingWindowPatchExtractor(input_img=reader, patch_size=(model.tile, model.tile),
                                     stride=(model.tile - model.overlap,) * 2, resolution=model.mpp, units="mpp",
                                     input_mask="otsu", min_mask_ratio=model.min_tissue, within_bound=False)
    b = np.asarray(ex.coordinate_list, dtype=np.int64).reshape(-1, 4)
    return b[np.lexsort((b[:, 0], b[:, 1]))]


class BandStitcher:
    """Weighted (Hann) blending of overlapping tiles that arrive in row order, written into a uint8 memmap."""

    def __init__(self, out: np.ndarray, tile: int):
        self.out, self.tile = out, tile
        h, w = out.shape[:2]
        self.w = w
        self.acc = np.zeros((tile, w, 3), np.float32)
        self.wt = np.zeros((tile, w), np.float32)
        self.top = 0                                      # output row of acc[0]
        win = np.hanning(tile + 2)[1:-1].astype(np.float32)
        self.window = np.outer(win, win)

    def _flush(self, upto: int):
        """Write finished rows [top, upto) to the output and slide the band down."""
        n = min(upto, self.out.shape[0]) - self.top
        if n <= 0:
            return
        k = min(n, self.tile)
        w = self.wt[:k]
        img = np.where(w[..., None] > 1e-8, self.acc[:k] / np.maximum(w, 1e-8)[..., None], 1.0)   # background white
        self.out[self.top:self.top + k] = np.rint(img * 255).astype(np.uint8)
        self.acc = np.roll(self.acc, -k, axis=0)
        self.wt = np.roll(self.wt, -k, axis=0)
        self.acc[-k:] = 0
        self.wt[-k:] = 0
        self.top += k
        if n > k:                                         # rows between tiles that no tile covered: background
            self.out[self.top:self.top + n - k] = 255
            self.top += n - k

    def add(self, x0: int, y0: int, pred: np.ndarray):
        """pred: [3, tile, tile] float in [0, 1] for the tile at (x0, y0)."""
        if y0 > self.top:
            self._flush(y0)
        t = self.tile
        ys, xs = max(0, -y0), max(0, -x0)
        ye, xe = min(t, self.out.shape[0] - y0), min(t, self.w - x0)
        if ye <= ys or xe <= xs:
            return
        r0 = y0 + ys - self.top
        win = self.window[ys:ye, xs:xe]
        self.acc[r0:r0 + ye - ys, x0 + xs:x0 + xe] += pred[:, ys:ye, xs:xe].transpose(1, 2, 0) * win[..., None]
        self.wt[r0:r0 + ye - ys, x0 + xs:x0 + xe] += win

    def finish(self):
        self._flush(self.out.shape[0])


def write_pyramid(img: np.ndarray, path: Path, mpp: float, min_size: int = 512):
    """Pyramidal OME-TIFF (RGB, JPEG q87, 512 px tiles), each level half the previous one. Large levels are built in
    disk-backed arrays next to `path`, so memory stays small. Returns the smallest level (for a thumbnail)."""
    import tifffile

    levels, temps = [img], []
    while min(levels[-1].shape[:2]) > 2 * min_size:
        a = levels[-1]
        h, w = a.shape[0] // 2, a.shape[1] // 2
        if h * w > 64_000_000:
            temps.append(path.with_suffix(f".l{len(levels)}.npy"))
            down = np.lib.format.open_memmap(temps[-1], mode="w+", dtype=np.uint8, shape=(h, w, 3))
        else:
            down = np.empty((h, w, 3), np.uint8)
        for y in range(0, h, 2048):                       # in strips, so a level never needs another copy in memory
            y1 = min(h, y + 2048)
            strip = a[2 * y:2 * y1, :2 * w].astype(np.uint16)
            down[y:y1] = ((strip[0::2, 0::2] + strip[1::2, 0::2] + strip[0::2, 1::2] + strip[1::2, 1::2] + 2) // 4)
        levels.append(down)
    opts = dict(tile=(512, 512), compression="jpeg", compressionargs={"level": 87}, photometric="rgb")
    try:
        _write(tifffile, path, levels, mpp, opts)
    finally:
        thumb = np.array(levels[-1])
        del levels
        for f in temps:
            f.unlink(missing_ok=True)
    return thumb


def _write(tifffile, path, levels, mpp, opts):
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tw:   # temp name has no .ome suffix, so say it
        tw.write(levels[0], subifds=len(levels) - 1, resolution=(1e4 / mpp, 1e4 / mpp), resolutionunit="CENTIMETER",
                 metadata={"axes": "YXS", "PhysicalSizeX": mpp, "PhysicalSizeY": mpp, "PhysicalSizeXUnit": "µm",
                           "PhysicalSizeYUnit": "µm"}, **opts)
        for i, lv in enumerate(levels[1:], 1):
            tw.write(lv, subfiletype=1, resolution=(1e4 / (mpp * 2 ** i),) * 2, resolutionunit="CENTIMETER", **opts)


def run_virtual_stain(model, slide, out_dir: Path, batch_size: int, workers: int, fallback_mpp: float | None):
    """Virtually stain one slide; writes <out_dir>/<slide>.ome.tiff and <slide>_thumb.jpg."""
    from PIL import Image

    reader = open_slide(slide.path, fallback_mpp)
    w, h = size_at(reader, model.mpp)
    bounds = tile_bounds(reader, model)
    if not len(bounds):
        raise ValueError("no tissue found (Otsu mask)")
    tmp = out_dir / f".{slide.id}.canvas"
    canvas = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(h, w, 3))
    try:
        stitch = BandStitcher(canvas, model.tile)
        ds = TileDataset(slide.path, bounds, TileSpec(model.tile, model.mpp), fallback_mpp)
        for tiles, idx in loader(ds, batch_size, workers, desc=f"stain {slide.id}"):
            out = model(tiles).float().cpu().numpy()
            for o, i in zip(out, idx.tolist()):
                stitch.add(int(bounds[i, 0]), int(bounds[i, 1]), o)
        stitch.finish()
        canvas.flush()
        path = out_dir / f"{slide.id}.ome.tiff"
        part = out_dir / f".{slide.id}.ome.tiff.part"
        thumb = write_pyramid(canvas, part, model.mpp)
        part.replace(path)
        im = Image.fromarray(thumb)
        im.thumbnail((1600, 1600))
        im.save(out_dir / f"{slide.id}_thumb.jpg", quality=88)
    finally:
        del canvas
        tmp.unlink(missing_ok=True)
    return {"tiles": int(len(bounds)), "width": int(w), "height": int(h), "mpp": model.mpp}
