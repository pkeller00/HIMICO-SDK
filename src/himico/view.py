"""`himico view <workspace>`: open a finished run in the TIAToolbox slide viewer.

Builds <workspace>/viewer/ with
    slides/     links to the original slide files (nothing is copied)
    overlays/   per slide: <stem>_insight_risk.db   INSIGHT local risk per patch (colour by `risk`, 0-1)
                           <stem>_tissue_types.db   CRC tissue class per tile (colour by `max_class`)
                           <stem>_qc_artefacts.png  GrandQC artefact map (fold, dark spot, pen, edge/bubble, out of focus)
and starts `tiatoolbox visualize` on it. In the viewer, pick a slide on the left, then add overlays from the list;
overlays are matched to slides by file name.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .stages.mask import QC_COLOURS


def _link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:                       # e.g. Windows without symlink rights: fall back to a hard link
        os.link(src, dst)


def prepare(workspace: str | Path) -> Path:
    ws = Path(workspace)
    if not (ws / "config.yaml").exists():
        raise FileNotFoundError(f"{ws} is not a HIMICO workspace (no config.yaml)")
    cfg = yaml.safe_load((ws / "config.yaml").read_text())
    model = cfg.get("model", "insight-crc")
    out = ws / "viewer"
    sdir, odir = out / "slides", out / "overlays"
    sdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ws / "cohort.csv")
    n_ov = 0
    for r in man.itertuples(index=False):
        src = Path(r.slide_path)
        if not src.exists():
            continue
        _link(src, sdir / src.name)
        if src.suffix.lower() == ".mrxs" and (src.parent / src.stem).is_dir():   # MRXS keeps its data in a folder
            _link(src.parent / src.stem, sdir / src.stem)
        stem = Path(src.name).stem              # the viewer matches overlays by the slide file's stem
        risk = ws / "predictions" / model / "overlays" / f"{r.slide_id}.db"
        if risk.exists():
            _link(risk, odir / f"{stem}_insight_risk.db")
            n_ov += 1
        tissue = ws / "tissue" / f"{r.slide_id}.db"
        if tissue.exists():
            _link(tissue, odir / f"{stem}_tissue_types.db")
            n_ov += 1
        qc = ws / "masks" / "qc" / f"{r.slide_id}_classes.png"
        target = odir / f"{stem}_qc_artefacts.png"
        if qc.exists() and (not target.exists() or target.stat().st_mtime < qc.stat().st_mtime):
            cls = np.array(Image.open(qc))
            rgba = np.zeros((*cls.shape, 4), np.uint8)
            rgba[..., :3] = QC_COLOURS[np.clip(cls, 0, 7)]
            # show only artefacts: normal tissue, background and padding are transparent
            rgba[..., 3] = np.where(np.isin(cls, [2, 3, 4, 5, 6]), 200, 0)
            Image.fromarray(rgba, "RGBA").save(target)
        if target.exists():
            n_ov += 1
    # viewer settings (keys read by the TIAToolbox viewer): colour by risk with blue-to-red, fixed tissue colours
    (odir / "config.json").write_text(json.dumps({
        "default_cprop": "risk",
        "UI_settings": {"mapper": "coolwarm", "max_scale": 1000},  # draw patches even at whole-slide zoom
        "opts": {"hover_on": 1},
        "color_dict": {"TUM": [220, 38, 38], "NORM": [51, 160, 44], "LYM": [106, 61, 154], "MUS": [247, 129, 191],
                       "STR": [255, 127, 0], "DEB": [128, 128, 128], "MUC": [102, 194, 219], "ADI": [242, 217, 51]},
    }, indent=1))
    (out / "README.txt").write_text(
        "Open with: himico view <workspace>  (or: tiatoolbox visualize --slides slides --overlays overlays)\n"
        "Overlays per slide:\n"
        "  *_insight_risk.db   INSIGHT local risk per patch; colour by 'risk' (0 = lower, 1 = higher risk)\n"
        "  *_tissue_types.db   CRC tissue type per tile; colour by 'max_class'\n"
        "  *_qc_artefacts.png  GrandQC artefacts: orange fold, green dark spot, red pen, pink edge/bubble, violet out of focus\n")
    return out


def _free_port(start: int) -> int:
    """First port from `start` that nothing is listening on (e.g. 5000/5006 are often taken by SSH tunnels)."""
    import socket

    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port found from {start}")


def view(workspace: str | Path, port: int = 5006, noshow: bool = False):
    out = prepare(workspace)
    port = _free_port(port)
    tile_port = _free_port(5000 if port != 5000 else 5001)
    if tile_port == port:
        tile_port = _free_port(port + 1)
    # TIAToolbox starts `bokeh serve` from PATH: make sure it is this environment's, not another Python's
    env = {**os.environ, "TIATOOLBOX_TILESERVER_PORT": str(tile_port),
           "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}
    cmd = [sys.executable, "-m", "tiatoolbox", "visualize", "--slides", str(out / "slides"),
           "--overlays", str(out / "overlays"), "--port", str(port)]
    if noshow:
        cmd.append("--noshow")
    print(f"viewer files: {out}\nTIAToolbox viewer: http://localhost:{port}/bokeh_app  (tiles on port {tile_port}; Ctrl-C to stop)")
    return subprocess.call(cmd, env=env)
