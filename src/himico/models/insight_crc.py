"""INSIGHT for stage II/III colorectal cancer (Keller et al., arXiv:2512.22262).

Four cross-validation fold models (TransformerConv GNN on UNI features); the patient score is their mean. The model
outputs a survival score (higher = better), so risk = -score and patients with risk >= 0.04099833 (training-set
threshold) are "High" risk. Local risk per patch = 1 - (node survival output normalised with each fold's 2nd-98th
percentile min-max), averaged over folds and clipped to [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import weights
from ..registry import register_model
from .base import Prediction, Predictor

THRESHOLD = 0.04099833333108103


class _LinearEmbed(torch.nn.Module):
    def __init__(self, i, o, dropout):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Linear(i, o), torch.nn.Dropout(dropout))
        self.norm = torch.nn.LayerNorm(o)

    def forward(self, x):
        return self.norm(self.conv(x))


class _Transformer(torch.nn.Module):
    def __init__(self, i, o, skip, dropout):
        super().__init__()
        from torch_geometric.nn import TransformerConv

        self.skip_conn = skip
        self.conv = TransformerConv(i, o, dropout=dropout, beta=True)
        self.norm = torch.nn.LayerNorm(o)

    def forward(self, x, ei):
        x = x + self.conv(x, ei) if self.skip_conn else self.conv(x, ei)
        return F.gelu(self.norm(x))


class _Stem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_LinearEmbed(1024, 32, 0.4), _Transformer(32, 32, True, 0.4)])

    def forward(self, x, ei):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x, ei)
        return x


class _Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import AttentionalAggregation

        self.layers = torch.nn.ModuleList([_Transformer(32, 16, False, 0.3)])
        self.cls = torch.nn.Linear(16, 1)
        self.pool = AttentionalAggregation(torch.nn.Linear(16, 1), torch.nn.Linear(16, 1))

    def forward(self, x, ei):
        for layer in self.layers:
            x = layer(x, ei)
        out = self.pool(x)
        g = self.pool.gate_nn(x)
        e = torch.exp(g - g.max())
        return (e / e.mean()) * self.pool.nn(x), out


class _Loss(torch.nn.Module):     # only to match saved state-dict keys
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterDict({"surv_time": torch.nn.Parameter(torch.ones(1))})


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_model = _Stem()
        self.branch_models = torch.nn.ModuleDict({"surv_time": _Branch()})
        self.loss_weighter = _Loss()

    def forward(self, x, ei):
        return self.branch_models["surv_time"](self.stem_model(x, ei), ei)


@register_model("insight-crc")
class InsightCRC(Predictor):
    features = "uni"
    dthresh = 4000.0
    local_name = "INSIGHT risk"
    description = "Overall-survival risk for stage II/III colorectal cancer (4-fold ensemble)"
    paper = "https://arxiv.org/abs/2512.22262"
    folds = 4
    weights = tuple(f"insight/fold{k}/{f}" for k in range(1, 5) for f in ("model.safetensors", "minmax.csv"))

    def __init__(self, device: torch.device, weights_dir: str | None = None):
        super().__init__(device, weights_dir)
        from safetensors.torch import load_file

        self.nets = []
        for k in range(1, self.folds + 1):
            net = _Net()
            net.load_state_dict(load_file(weights.resolve(f"insight/fold{k}/model.safetensors", weights_dir)))
            mm = pd.read_csv(weights.resolve(f"insight/fold{k}/minmax.csv", weights_dir)).set_index("Unnamed: 0")
            lo, hi = (float(v) for v in mm.loc["surv_time"].iloc[:2])
            self.nets.append((net.to(device).eval(), lo, hi))

    @torch.no_grad()
    def predict(self, graph) -> Prediction:
        x = graph.x.float().to(self.device)
        ei = graph.edge_index.long().to(self.device)
        scores, local = [], []
        for net, lo, hi in self.nets:
            node, out = net(x, ei)
            scores.append(float(out.squeeze()))
            local.append(1.0 - (node.squeeze(1).cpu().numpy() - lo) / ((hi - lo) or 1.0))
        score = float(np.mean(scores))
        risk = -score
        return Prediction(patient=str(getattr(graph, "id", "")), score=score, risk=risk,
                          group="High" if risk >= THRESHOLD else "Low", folds=scores,
                          local=np.clip(np.mean(local, axis=0), 0, 1).astype(np.float32),
                          extra={"n_patches": int(graph.x.shape[0]), "n_slides": len(getattr(graph, "slideOrder", []) or [1])})
