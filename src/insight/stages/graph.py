"""Stage 4: slide graphs and patient graphs.

Slide graph (as SlideGraph+ / the INSIGHT pipeline): nodes are patches, edges join Delaunay neighbours closer than
`dthresh` pixels (top-left corners at the feature resolution).

Reproducibility: patches lie on a square grid, where Delaunay triangulation is not unique (either diagonal of a square
is valid). SciPy <= 1.12 (as used to build the INSIGHT training graphs) and SciPy >= 1.13 break these ties differently, so about 12%
of edges, nearly all 45-degree diagonals, differ. Effect on the public TCGA example: risk 0.23708 vs 0.23777, per-patch
Spearman 0.9965. SciPy <= 1.12 cannot be used with TIAToolbox 2 (it needs NumPy 2); the SciPy version is recorded in
environment.json.

Patient graph: the slide graphs of one patient are concatenated into one graph (no edges between slides), so the model
sees the patient as a whole. As in the INSIGHT training pipeline, each later slide's coordinates are offset by the
number of nodes before it; `slideOrder` names the slides and `slide_ptr` gives each slide's node range, so per-slide
maps can be recovered exactly.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree


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
        for simplex in Delaunay(coords).simplices:
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
