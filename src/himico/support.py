"""`himico verify` (known-answer checks before an expensive run) and `himico bundle` (privacy-safe support file)."""
from __future__ import annotations

import hashlib
import io
import json
import re
import secrets
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

# Reference result for the public TCGA example graph (github.com/pkeller00/INSIGHT, data/graphs/TCGA-4N-A93T.g).
# Fold 4 held this patient out in cross-validation; its stored out-of-fold risk is 0.336644.
REFERENCE = {"model": "insight-crc", "risk": 0.23708, "group": "High", "fold4_risk": 0.33664}
TOLERANCE = 2e-3          # devices (CPU / CUDA / MPS) differ slightly in floating point


def verify(device: str = "auto", weights_dir: str | None = None, backend: str | None = "auto", quick: bool = False,
           model: bool = True):
    """Returns a list of (check, ok, detail). Model check uses a known answer (locally or on the server, as chosen by
    `backend`); preprocessing networks always run locally on a synthetic tile (shapes, finite values, determinism)."""
    from . import weights
    from .models.remote import resolve_backend
    from .registry import get_model
    from .safe_graph import load_graph
    from .wsi import device_for

    dev = device_for(device)
    backend = resolve_backend(backend, get_model(REFERENCE["model"]), weights_dir) if model else None
    results = []

    def check(name, fn):
        t = time.perf_counter()
        try:
            detail = fn()
            results.append((name, True, f"{detail} ({time.perf_counter() - t:.1f}s)"))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))

    def model_check():
        g = load_graph(Path(weights.resolve("verify/TCGA-4N-A93T.g", weights_dir)).read_bytes())
        if backend:
            from .models.remote import RemotePredictor
            m = RemotePredictor(backend, REFERENCE["model"], get_model(REFERENCE["model"]))
        else:
            m = get_model(REFERENCE["model"])(dev, weights_dir)
        p = m.predict(g)
        if abs(p.risk - REFERENCE["risk"]) > TOLERANCE or p.group != REFERENCE["group"]:
            raise AssertionError(f"risk {p.risk:.5f} ({p.group}); expected {REFERENCE['risk']} ({REFERENCE['group']})")
        if abs(-p.folds[3] - REFERENCE["fold4_risk"]) > TOLERANCE:
            raise AssertionError(f"fold 4 risk {-p.folds[3]:.5f}; expected {REFERENCE['fold4_risk']}")
        return f"risk {p.risk:.5f} {p.group}, matches reference {REFERENCE['risk']}" + (f" via {backend}" if backend else "")

    if model:
        check(f"INSIGHT model on example TCGA graph ({'server' if backend else dev})", model_check)
    if quick:
        return results

    rng = np.random.default_rng(0)
    tile = (rng.random((2, 512, 512, 3)) * 255).astype(np.uint8)

    def grandqc():
        from .stages.mask import GrandQC
        qc = GrandQC(dev, 1.5, weights_dir)
        a, b = qc._segment(qc.qc, tile), qc._segment(qc.qc, tile)
        t = qc._segment(qc.td, tile)
        assert a.shape == (2, 512, 512) and np.array_equal(a, b) and t.shape == (2, 512, 512)
        return "tissue detection + artefact QC load and run deterministically"

    def tissue():
        from .stages.tissue import CRCTissue
        m = CRCTissue(dev, weights_dir)
        p = m(torch.from_numpy(tile[:, :224, :224]))
        assert p.shape == (2, 8) and torch.isfinite(p).all() and torch.allclose(p.sum(1), torch.ones(2), atol=1e-4)
        return "CRC tissue classifier loads and returns 8 class probabilities"

    def features():
        from .registry import get_features
        ext = get_features("uni")(dev, weights_dir)
        f = ext(torch.from_numpy(tile))
        assert f.shape == (2, 1024) and torch.isfinite(f).all()
        return "UNI loads and returns 1024-d features"

    check("GrandQC", grandqc)
    check("CRC tissue classifier", tissue)
    check("UNI feature extractor", features)
    return results


# ── support bundle ───────────────────────────────────────────────────────────────────────────────────────────────────
_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[^\s\"',:;|<>]+[/\\])+[^\s\"',:;|<>]*")


def bundle(workspace: str | Path, out: str | Path | None = None) -> Path:
    """Pack config, environment, report, status and logs into a .tar.gz with slide/patient IDs and file paths replaced
    by pseudonyms (stable within the bundle, random salt per bundle). No images, features or predictions are included.
    Review the bundle before sharing it."""
    import pandas as pd

    ws = Path(workspace)
    if not (ws / "config.yaml").exists():
        raise FileNotFoundError(f"{ws} is not a HIMICO workspace")
    salt = secrets.token_hex(8)
    tokens: dict[str, str] = {}

    def pseudo(kind, value):
        return f"{kind}_{hashlib.sha256((salt + value).encode()).hexdigest()[:10]}"

    if (ws / "cohort.csv").exists():
        man = pd.read_csv(ws / "cohort.csv")
        for v in man.get("slide_path", []):
            tokens[str(v)] = pseudo("path", str(v))
        for v in man.get("slide_id", []):
            tokens[str(v)] = pseudo("slide", str(v))
        for v in man.get("patient_id", []):
            tokens.setdefault(str(v), pseudo("patient", str(v)))
    order = sorted(tokens, key=len, reverse=True)
    rx = re.compile("|".join(re.escape(t) for t in order)) if order else None

    def scrub(text: str) -> str:
        if rx:
            text = rx.sub(lambda m: tokens[m.group(0)], text)
        return _PATH.sub(lambda m: pseudo("path", m.group(0)), text)

    files = [ws / "config.yaml", ws / "environment.json", ws / "report.csv", ws / "cohort.csv"]
    files += sorted((ws / "logs").glob("*.jsonl")) + sorted((ws / "status").glob("*.json"))
    out = Path(out) if out else ws / f"himico_support_{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            if not f.exists():
                continue
            data = scrub(f.read_text(errors="replace")).encode()
            name = scrub(str(f.relative_to(ws)))
            info = tarfile.TarInfo(f"himico_support/{name}")
            info.size, info.mtime = len(data), int(time.time())
            tar.addfile(info, io.BytesIO(data))
        readme = ("HIMICO support bundle. Slide IDs, patient IDs and file paths were replaced by pseudonyms "
                  "(consistent within this bundle only). No images, features or predictions are included.\n").encode()
        info = tarfile.TarInfo("himico_support/README.txt")
        info.size, info.mtime = len(readme), int(time.time())
        tar.addfile(info, io.BytesIO(readme))
    return out
