"""Safe loading of patient graphs (.g files) received from other people.

.g files are torch.save'd pickles, and unpickling untrusted data can execute code. They are therefore opened only with
torch.load(weights_only=True) and an allow-list of the PyTorch Geometric classes a graph needs; anything else is
refused before model code runs. The graph is then checked for type, shape, range and size.
"""
from __future__ import annotations

import io
import zipfile

import torch

MAX_BYTES = 300 * 1024 * 1024
MAX_NODES = 200_000
MAX_EDGES = 4_000_000


class GraphError(ValueError):
    pass


def _safe_globals():
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    return [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]


def load_graph(raw: bytes, in_channels: int = 1024):
    """Parse and validate an untrusted .g file; returns a torch_geometric Data object."""
    from torch_geometric.data.data import Data

    if len(raw) > MAX_BYTES:
        raise GraphError(f"file is larger than {MAX_BYTES // 2**20} MB")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise GraphError("not a torch.save file")
    try:
        with torch.serialization.safe_globals(_safe_globals()):
            g = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as e:
        raise GraphError(f"could not be opened safely ({type(e).__name__}); only PyTorch Geometric graphs are accepted") from None
    if not isinstance(g, Data):
        raise GraphError("file does not contain a torch_geometric Data object")
    x, ei = getattr(g, "x", None), getattr(g, "edge_index", None)
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[1] != in_channels or not x.is_floating_point():
        raise GraphError(f"x must be a float tensor of shape [N, {in_channels}]")
    n = x.shape[0]
    if not 1 <= n <= MAX_NODES:
        raise GraphError(f"graph must have between 1 and {MAX_NODES:,} nodes")
    if not isinstance(ei, torch.Tensor) or ei.dim() != 2 or ei.shape[0] != 2 or ei.dtype not in (torch.int64, torch.int32):
        raise GraphError("edge_index must be an integer tensor of shape [2, E]")
    if ei.shape[1] > MAX_EDGES:
        raise GraphError(f"graph must have at most {MAX_EDGES:,} edges")
    if ei.numel() and (int(ei.min()) < 0 or int(ei.max()) >= n):
        raise GraphError("edge_index refers to nodes that do not exist")
    if not torch.isfinite(x).all():
        raise GraphError("x contains NaN or infinite values")
    ptr = getattr(g, "slide_ptr", None)
    if ptr is not None and (not isinstance(ptr, torch.Tensor) or ptr.dim() != 1 or int(ptr[-1]) != n):
        raise GraphError("slide_ptr is inconsistent with the number of nodes")
    g.x, g.edge_index = x.float(), ei.long()
    return g


def dump_graph(g) -> bytes:
    buf = io.BytesIO()
    torch.save(g, buf)
    return buf.getvalue()
