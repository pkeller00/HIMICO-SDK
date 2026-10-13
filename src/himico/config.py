"""Pipeline configuration: one object (and one YAML file) describes a whole run."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MaskConfig:
    """Stage 1. Where the slide-level tissue/QC mask comes from."""

    source: str = "grandqc"          # "grandqc" or a folder of <slide>.png masks you provide
    qc_mpp: float = 1.5              # GrandQC artefact model resolution: 1.0, 1.5 or 2.0
    keep_classes: list[int] = field(default_factory=lambda: [1, 2, 5, 6])  # normal, fold, edge/bubble, out-of-focus


@dataclass
class TissueConfig:
    """Stage 2. CRC tissue-type classification; the selected types define where features are taken."""

    model: str = "crc-tissue-vit"
    keep: list[str] = field(default_factory=lambda: ["TUM", "STR", "LYM"])
    tile: int = 224
    mpp: float = 0.5
    mask_mpp: float = 8.0
    min_mask_fraction: float = 0.4
    write_overlay: bool = True        # TIAToolbox .db with per-tile class probabilities


@dataclass
class FeatureConfig:
    """Stage 3. Patch embeddings."""

    model: str = "uni"
    tile: int = 512
    mpp: float = 0.5
    min_mask_fraction: float = 0.4


@dataclass
class GraphConfig:
    """Stage 4. Slide graphs (Delaunay, edges up to `dthresh` pixels at the feature resolution), merged per patient."""

    dthresh: float = 4000.0


@dataclass
class PipelineConfig:
    output: str = "himico_runs"
    experiment: str | None = None
    run: int | None = None
    variant: str | None = None
    description: str = ""
    model: str = "insight-crc"
    device: str = "auto"               # auto | cuda | cuda:1 | mps | cpu
    batch_size: int = 64
    workers: int = 4
    mpp: float | None = None           # fallback microns-per-pixel for slides that lack it
    weights_dir: str | None = None     # local folder with weights (otherwise Hugging Face)
    backend: str | None = "auto"       # where the model runs: auto | himico | local | URL of a `himico serve` server
    mask: MaskConfig = field(default_factory=MaskConfig)
    tissue: TissueConfig = field(default_factory=TissueConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    # ── workspace ──────────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def workspace(self) -> Path:
        p = Path(self.output)
        if self.experiment:
            p = p / self.experiment
        if self.run is not None:
            p = p / f"run_{self.run}"
        if self.variant:
            p = p / self.variant
        return p

    # ── (de)serialisation ──────────────────────────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        d = dict(d or {})
        sections = {"mask": MaskConfig, "tissue": TissueConfig, "features": FeatureConfig, "graph": GraphConfig}
        kw = {}
        for k, v in d.items():
            if k in sections:
                kw[k] = sections[k](**(v or {}))
            elif k in {f.name for f in dataclasses.fields(cls)}:
                kw[k] = v
            else:
                raise ValueError(f"unknown config key: {k!r}")
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})


def template_yaml() -> str:
    """A commented starter config for `himico init`."""
    return """\
# HIMICO pipeline configuration. Every field is optional; these are the defaults.
output: himico_runs          # workspace = <output>/<experiment>/run_<run>/<variant>
experiment: my_cohort
run: 1
description: ""
model: insight-crc            # see `himico models`
device: auto                  # auto | cuda | cuda:1 | mps | cpu
batch_size: 64
workers: 4
mpp: null                     # fallback microns-per-pixel if a slide does not record it
weights_dir: null             # folder with local weights; otherwise downloaded from Hugging Face
backend: auto                 # auto: local model weights if you have them, else the HIMICO server (API key:
                              #   `himico login`); himico | local | http://gpu-server:8080 (a `himico serve` server)

mask:
  source: grandqc             # or a folder of <slide>.png masks you made yourself
  qc_mpp: 1.5
  keep_classes: [1, 2, 5, 6]  # GrandQC: normal, fold, edge/air bubble, out-of-focus

tissue:
  keep: [TUM, STR, LYM]       # CRC tissue types used for features (tumour, stroma, lymphocytes)
  write_overlay: true

features:
  model: uni                  # UNI needs your own approved access: huggingface.co/MahmoodLab/UNI

graph:
  dthresh: 4000
"""
