"""Stage 4: slide graphs and patient graphs.

Slide graph (as SlideGraph+ / the INSIGHT pipeline): nodes are patches, edges join Delaunay neighbours closer than
`dthresh` pixels (top-left corners at the feature resolution).

Reproducibility: patches lie on a square grid, where Delaunay triangulation is not unique (either diagonal of a square
is valid) and the choice depends on floating-point rounding. INSIGHT's training graphs were built with Qhull 2019.1
compiled without fused multiply-add; newer SciPy wheels contract multiply-adds and pick different diagonals (about 12%
of edges). The SDK therefore triangulates with its own copy of that Qhull (insight._qhull_legacy, compiled with
-ffp-contract=off), which reproduces the training edges exactly. If the extension is missing it falls back to SciPy and
warns.

Patient graph: the slide graphs of one patient are concatenated into one graph (no edges between slides), so the model
sees the patient as a whole. As in the INSIGHT training pipeline, each later slide's coordinates are offset by the
number of nodes before it; `slideOrder` names the slides and `slide_ptr` gives each slide's node range, so per-slide
maps can be recovered exactly.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree

log = logging.getLogger("insight")
try:
    from .. import _qhull_legacy
except ImportError:      # pragma: no cover  (extension not built)
    _qhull_legacy = None
_warned = False


def delaunay_simplices(coords: np.ndarray) -> np.ndarray:
    """[M, 3] triangles: vendored Qhull 2019.1 (training-identical) or SciPy as a fallback."""
    global _warned
    pts = np.ascontiguousarray(coords, dtype=np.float64)
    if _qhull_legacy is not None:
        return np.frombuffer(_qhull_legacy.delaunay(pts.tobytes()), dtype=np.int32).reshape(-1, 3)
    if not _warned:
        log.warning("insight._qhull_legacy is not built; using SciPy's Delaunay, whose ties on the patch grid can "
                    "differ from the training graphs (reinstall insight-crc with a C compiler available)")
        _warned = True
    return Delaunay(pts).simplices


def connect_patches(coords: np.ndarray, dthresh: float) -> np.ndarray:
    """[2, E] edge index, both directions, row-major order (same edge set as the dense-matrix original)."""
    n = len(coords)
    if n < 2:
        return np.zeros((2, 0), np.int64)
    pairs: set[tuple[int, int]] = set()
    if n < 3 or np.linalg.matrix_rank(coords - coords.mean(0)) < 2:     # Delaunay needs a 2-D point set
        tree = KDTree(coords)
        for i, j in tree.query_pairs(dthresh):
            pairs.update({(i, j), (j, i)})
    else:
        nbrs: dict[int, set[int]] = defaultdict(set)
        for simplex in delaunay_simplices(coords):
            for i in simplex:
                nbrs[int(i)].update(int(j) for j in simplex if j != i)
        for i, js in nbrs.items():
            js = np.fromiter(js, np.int64)
            close = js[np.linalg.norm(coords[js] - coords[i], axis=1) <= dthresh]
            for j in close:
                pairs.update({(i, int(j)), (int(j), i)})
    if not pairs:
        return np.zeros((2, 0), np.int64)
    return np.array(sorted(pairs), np.int64).T


def slide_graph(features_npz: Path, dthresh: float):
    from torch_geometric.data import Data

    z = np.load(features_npz)
    coords = z["bounds"][:, :2].astype(np.float32)
    ei = connect_patches(coords.astype(np.float64), dthresh)
    return Data(x=torch.from_numpy(z["features"]).float(), edge_index=torch.from_numpy(ei).long(),
                coords=torch.from_numpy(coords))


def patient_graph(patient: str, slide_ids: list[str], graphs: list):
    from torch_geometric.data import Data

    xs, eis, cs, ptr, offset = [], [], [], [0], 0
    for g in graphs:
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        cs.append(g.coords + offset)         # as in the training pipeline (kept for identical graphs)
        offset += g.x.shape[0]
        ptr.append(offset)
    G = Data(x=torch.cat(xs), edge_index=torch.cat(eis, 1), coords=torch.cat(cs))
    G.id = patient
    G.patient = patient
    G.slideOrder = list(slide_ids)
    G.slide_ptr = torch.tensor(ptr, dtype=torch.long)
    G.y = {}
    return G
