"""Interface every prediction model implements, so models can be swapped by name in the config."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Prediction:
    """Result for one patient graph."""

    patient: str
    score: float                                  # model output
    risk: float                                   # higher = worse prognosis
    group: str | None = None                      # e.g. "High" / "Low"
    folds: list[float] = field(default_factory=list)
    local: np.ndarray | None = None               # [N] per-patch value in [0, 1] for maps, aligned with graph nodes
    extra: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        r = {"patient_id": self.patient, "score": self.score, "risk": self.risk, "risk_group": self.group}
        r.update({f"fold{i + 1}_score": s for i, s in enumerate(self.folds)})
        r.update(self.extra)
        return r


class Predictor:
    """Subclass and register with `@register_model("name")`.

    Class attributes declare what the model needs from earlier stages, so the pipeline can check a config:
      features   name of the registered feature extractor the model was trained on
      dthresh    graph edge distance used in training (None: use the config value)
      local_name label for the per-patch map (e.g. "INSIGHT risk")
      weights    weight files (keys in himico.weights.FILES); if they are not on this machine, backend "auto" sends
                 graphs to the HIMICO server instead
    """

    name = "base"
    task = "risk"                                  # patient-level prediction from a graph of patch features
    stages = ("mask", "tissue", "features", "graphs", "predict")
    features = "uni"
    dthresh: float | None = None
    local_name = "local value"
    weights: tuple[str, ...] = ()
    description = ""
    paper = ""

    def __init__(self, device: torch.device, weights_dir: str | None = None):
        self.device = device

    def predict(self, graph) -> Prediction:  # pragma: no cover
        raise NotImplementedError


class VirtualStainer:
    """Image-to-image model applied tile by tile across the slide (e.g. H&E -> virtual multiplex IHC).

    Subclass and register with `@register_model("name")`. The pipeline reads `tile` x `tile` tiles at `mpp` with
    `overlap` pixels, keeping tiles whose tissue fraction (Otsu mask) is at least `min_tissue`, calls the model on
    batches of uint8 [B, H, W, 3] tiles and blends the outputs ([B, 3, H, W] in [0, 1]) into one pyramidal OME-TIFF.
    """

    name = "base"
    task = "virtual-stain"
    stages = ("stain",)
    weights: tuple[str, ...] = ()
    description = ""
    paper = ""
    tile = 512
    mpp = 0.5
    overlap = 64
    min_tissue = 0.2

    def __init__(self, device: torch.device, weights_dir: str | None = None):
        self.device = device

    def __call__(self, tiles: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
