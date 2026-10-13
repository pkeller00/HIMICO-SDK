"""Checks that the SDK reproduces the original INSIGHT pipeline pieces."""
import os
from collections import defaultdict

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial import KDTree

from himico.stages.graph import connect_patches

WEIGHTS = os.environ.get("INSIGHT_WEIGHTS")          # optional local weights folder; otherwise Hugging Face


def _example_graph():
    """Public TCGA example graph (downloaded once); skip the test if weights/access are unavailable."""
    from himico import weights
    try:
        return weights.resolve("verify/TCGA-4N-A93T.g", WEIGHTS)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"example graph unavailable: {e}")


def _original_connect(C, dthresh):
    """connectPatches from 5_extract_graphs/graphConstruction.py (SlideGraph+), dense W -> nonzero edges.
    Uses the same triangulation as the SDK, so this checks the neighbour/distance logic; the triangulation itself is
    checked against a real training graph in test_graph_edges_match_training_graph_exactly."""
    from himico.stages.graph import delaunay_simplices

    nb = defaultdict(set)
    for simplex in delaunay_simplices(C):
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            nb[idx] = nb[idx].union(other)
    W = np.zeros((C.shape[0], C.shape[0]))
    for n in nb:
        arr = np.array(list(nb[n]), dtype=np.int32)
        arr = arr[KDTree(C[arr, :]).query_ball_point(C[n], r=dthresh)]
        W[n, arr] = 1.0
        W[arr, n] = 1.0
    return np.array(np.nonzero(W > 0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_graph_edges_match_original(seed):
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(40), np.arange(30)), -1).reshape(-1, 2) * 512.0
    C = grid[rng.random(len(grid)) < 0.6]            # patchy tissue on a 512 px lattice
    C = np.concatenate([C, C[:20] + 20000])         # a second, distant fragment
    assert np.array_equal(connect_patches(C, 4000), _original_connect(C, 4000))


def test_insight_crc_matches_reference():
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    from himico.models import InsightCRC

    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        g = torch.load(_example_graph(), weights_only=True)
    try:
        model = InsightCRC(torch.device("cpu"), WEIGHTS)
    except Exception as e:  # pragma: no cover  (gated weights: needs approved access + hf auth login)
        pytest.skip(f"INSIGHT weights unavailable: {e}")
    pred = model.predict(g)
    # fold 4 held this patient out in cross-validation; its stored out-of-fold risk is 0.336644
    assert abs(-pred.folds[3] - 0.336644) < 1e-3
    assert abs(pred.risk - 0.23708) < 1e-4 and pred.group == "High"
    assert pred.local.shape == (g.x.shape[0],) and 0 <= pred.local.min() and pred.local.max() <= 1


def test_graph_edges_match_training_graph_exactly():
    """Public TCGA-4N-A93T DX1 slide: patch positions and the edges of the original INSIGHT graph. The vendored
    Qhull 2019.1 (no fused multiply-add) must reproduce them edge for edge, in order."""
    from himico.stages import graph

    if graph._qhull_legacy is None:
        pytest.skip("himico._qhull_legacy not built")
    z = np.load(Path(__file__).parent / "data" / "tcga_4n_a93t_dx1_graph.npz")
    e = connect_patches(z["coords"].astype(np.float64), 4000)
    assert np.array_equal(e, z["edge_index"].astype(np.int64))
