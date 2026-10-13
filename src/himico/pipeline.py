"""The HIMICO pipeline: one object configures and runs every stage over a cohort.

    from himico import Cohort, Pipeline

    cohort = Cohort.from_folder("/data/wsi", patient=r"^(TCGA-\\w\\w-\\w{4})")
    pipe = Pipeline(output="runs", experiment="tcga", run=1)
    preds = pipe.run(cohort)        # mask -> tissue -> features -> graphs -> predict; resumes where it stopped
    pipe.report()                   # per-slide status

Stages run cohort-wide one after another, so only one network is in memory at a time and every slide's result is
cached in the workspace. Re-running skips finished work and retries failed slides.

Workspace (<output>/<experiment>/run_<run>/<variant>/):
    config.yaml   cohort.csv   report.csv   logs/run_*.jsonl   status/<slide>.json
    masks/<slide>.png                     masks/qc/<slide>_{classes.png, overlay.jpg}
    tissue/<slide>.npz                    tissue/<slide>_mask.png    tissue/<slide>.db
    features/<extractor>/<slide>.npz      graphs/slides/<slide>.pt   graphs/patients/<patient>.g
    predictions/<model>/predictions.csv   predictions/<model>/{local/<patient>.npz, overlays/<slide>.db,
                                                                 heatmaps/<slide>.png, thumbnails/<slide>.jpg}
"""
from __future__ import annotations

import contextlib
import gc
import json
import logging
import platform
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from . import __version__
from .cohort import Cohort, Slide
from .config import PipelineConfig
from .log import setup as setup_logging
from .registry import get_features, get_model
from .wsi import device_for

log = logging.getLogger("himico")
STAGES = ("mask", "tissue", "features", "graphs", "predict")   # INSIGHT-style (graph) models; see Model.stages
# config fields that change results; a workspace refuses a different value for these
_RESULT_FIELDS = ("model", "mask", "tissue", "features", "graph", "mpp")


class ConfigMismatch(RuntimeError):
    pass


class Pipeline:
    def __init__(self, config: PipelineConfig | str | Path | None = None, **overrides):
        if isinstance(config, (str, Path)):
            config = PipelineConfig.from_yaml(config)
        cfg = (config or PipelineConfig()).to_dict()
        for k, v in overrides.items():
            if k in ("mask", "tissue", "features", "graph") and isinstance(v, dict):
                cfg[k].update(v)
            elif k in cfg:
                cfg[k] = v
            elif k == "masks":     # convenience: Pipeline(masks="/path/to/masks")
                cfg["mask"]["source"] = str(v)
            else:
                raise TypeError(f"unknown option {k!r}")
        self.config = PipelineConfig.from_dict(cfg)
        if not self.config.model:
            from .registry import models
            raise ValueError(f"choose a model, e.g. Pipeline(model=\"insight-crc\"); available: {sorted(models())}")
        model_cls = get_model(self.config.model)
        self.stages = tuple(model_cls.stages)
        if model_cls.task != "risk":
            self.device = device_for(self.config.device)
            self._log_path = None
            return
        if model_cls.features != self.config.features.model:
            raise ValueError(f"model {self.config.model!r} was trained on {model_cls.features!r} features, "
                             f"but features.model is {self.config.features.model!r}")
        if model_cls.dthresh is not None and model_cls.dthresh != self.config.graph.dthresh:
            log.warning("graph.dthresh=%s differs from %s used to train %s", self.config.graph.dthresh,
                        model_cls.dthresh, self.config.model)
        self.device = device_for(self.config.device)
        self._log_path = None

    # ── construction helpers ───────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_workspace(cls, workspace: str | Path, **overrides) -> "Pipeline":
        """Rebuild the exact pipeline that produced a workspace (cached results are reused)."""
        cfg = PipelineConfig.from_yaml(Path(workspace) / "config.yaml")
        return cls(cfg, **overrides)

    @property
    def workspace(self) -> Path:
        return self.config.workspace

    def _prepare(self, cohort: Cohort):
        ws = self.workspace
        ws.mkdir(parents=True, exist_ok=True)
        if self._log_path is None:
            self._log_path = setup_logging(ws)
        cfg_path = ws / "config.yaml"
        new = self.config.to_dict()
        if cfg_path.exists():
            import yaml
            old = yaml.safe_load(cfg_path.read_text())
            diff = [k for k in _RESULT_FIELDS if old.get(k) != new.get(k)]
            if diff:
                raise ConfigMismatch(f"{ws} was created with different settings for {diff}. Use a new `run` or "
                                     f"`variant`, or Pipeline.from_workspace({str(ws)!r}).")
        else:
            self.config.to_yaml(cfg_path)
            (ws / "environment.json").write_text(json.dumps(_environment(), indent=2))
        # cohort manifest (merged across runs/shards)
        man = ws / "cohort.csv"
        df = cohort.to_frame()
        if man.exists():
            df = pd.concat([pd.read_csv(man), df]).drop_duplicates("slide_id", keep="last")
        df.to_csv(man, index=False)

    # ── status bookkeeping ─────────────────────────────────────────────────────────────────────────────────────────
    def _status_path(self, key: str) -> Path:
        return self.workspace / "status" / f"{key}.json"

    def _status(self, key: str) -> dict:
        p = self._status_path(key)
        return json.loads(p.read_text()) if p.exists() else {}

    def _set_status(self, key: str, stage: str, ok: bool, seconds: float, info: dict | None = None, error: str = ""):
        p = self._status_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        st = self._status(key)
        st[stage] = {"status": "done" if ok else "failed", "seconds": round(seconds, 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     **({"info": info} if info else {}), **({"error": error} if error else {})}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(p)

    def _for_each(self, items: list, key: Callable, stage: str, done: Callable, work: Callable, retry_failed: bool,
                  label: str = "slide"):
        todo = [it for it in items if not (done(it) and self._status(key(it)).get(stage, {}).get("status") == "done")]
        if not retry_failed:
            todo = [it for it in todo if self._status(key(it)).get(stage, {}).get("status") != "failed"]
        log.info("%s: %d to do, %d already done", stage, len(todo), len(items) - len(todo), extra={"stage": stage})
        failures = 0
        for n, it in enumerate(todo, 1):
            t0 = time.perf_counter()
            k = key(it)
            try:
                info = work(it) or {}
                dt = time.perf_counter() - t0
                self._set_status(k, stage, True, dt, info)
                log.info("%s %d/%d %s %s (%.1fs)", stage, n, len(todo), label, k, dt,
                         extra={"stage": stage, label: k, "seconds": round(dt, 1), "status": "done"})
            except KeyboardInterrupt:
                raise
            except Exception as e:  # one bad slide must not stop a cohort
                failures += 1
                dt = time.perf_counter() - t0
                self._set_status(k, stage, False, dt, error=f"{type(e).__name__}: {e}")
                log.error("%s failed for %s %s: %s", stage, label, k, e, exc_info=log.isEnabledFor(logging.DEBUG),
                          extra={"stage": stage, label: k, "status": "failed"})
                log.debug(traceback.format_exc())
        if failures:
            log.warning("%s: %d %s(s) failed; see `himico report` and logs/", stage, failures, label)

    def _ready(self, s: Slide, stage: str) -> bool:
        return self._status(s.id).get(stage, {}).get("status") == "done"

    # ── stages ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def mask(self, cohort: Cohort, retry_failed: bool = True):
        """Stage 1: GrandQC tissue/artefact mask, or copy masks from the folder given as `mask.source`."""
        self._prepare(cohort)
        out = self.workspace / "masks"
        out.mkdir(parents=True, exist_ok=True)
        src = self.config.mask.source
        if src != "grandqc":
            from .stages.mask import use_folder_mask
            folder = Path(src)
            if not folder.is_dir():
                raise FileNotFoundError(f"mask.source must be 'grandqc' or a folder of masks; {folder} not found")
            self._for_each(cohort.slides, lambda s: s.id, "mask", lambda s: (out / f"{s.id}.png").exists(),
                           lambda s: use_folder_mask(folder, s, out), retry_failed)
            return
        from .stages.mask import GrandQC, run_grandqc
        with _model(lambda: GrandQC(self.device, self.config.mask.qc_mpp, self.config.weights_dir)) as qc:
            self._for_each(cohort.slides, lambda s: s.id, "mask", lambda s: (out / f"{s.id}.png").exists(),
                           lambda s: run_grandqc(qc, s, out, self.config.mask.keep_classes, self.config.mpp,
                                                 workers=self.config.workers, batch=min(16, self.config.batch_size)),
                           retry_failed)

    def tissue(self, cohort: Cohort, retry_failed: bool = True):
        """Stage 2: CRC tissue types; writes the tissue mask used for features."""
        self._prepare(cohort)
        from .stages.tissue import CRCTissue, run_tissue
        out = self.workspace / "tissue"
        out.mkdir(parents=True, exist_ok=True)
        slides = [s for s in cohort.slides if self._ready(s, "mask")]
        c = self.config
        with _model(lambda: CRCTissue(self.device, c.weights_dir)) as model:
            self._for_each(slides, lambda s: s.id, "tissue", lambda s: (out / f"{s.id}_mask.png").exists(),
                           lambda s: run_tissue(model, s, self.workspace / "masks" / f"{s.id}.png", out, c.tissue,
                                                c.batch_size, c.workers, c.mpp), retry_failed)

    def features(self, cohort: Cohort, retry_failed: bool = True):
        """Stage 3: patch embeddings with the configured extractor."""
        self._prepare(cohort)
        from .stages.features import run_features
        c = self.config
        out = self.workspace / "features" / c.features.model
        out.mkdir(parents=True, exist_ok=True)
        slides = [s for s in cohort.slides if self._ready(s, "tissue")]
        with _model(lambda: get_features(c.features.model)(self.device, c.weights_dir)) as ext:
            self._for_each(slides, lambda s: s.id, "features", lambda s: (out / f"{s.id}.npz").exists(),
                           lambda s: run_features(ext, s, self.workspace / "tissue" / f"{s.id}_mask.png", out,
                                                  c.features, c.batch_size, c.workers, c.mpp), retry_failed)

    def graphs(self, cohort: Cohort, retry_failed: bool = True):
        """Stage 4: slide graphs, then one patient graph per patient (all of its slides must have features)."""
        self._prepare(cohort)
        from .stages.graph import patient_graph, slide_graph
        c = self.config
        fdir = self.workspace / "features" / c.features.model
        sdir, pdir = self.workspace / "graphs" / "slides", self.workspace / "graphs" / "patients"
        sdir.mkdir(parents=True, exist_ok=True)
        pdir.mkdir(parents=True, exist_ok=True)
        slides = [s for s in cohort.slides if self._ready(s, "features")]

        def one(s):
            g = slide_graph(fdir / f"{s.id}.npz", c.graph.dthresh)
            torch.save(g, sdir / f"{s.id}.pt")
            return {"nodes": int(g.x.shape[0]), "edges": int(g.edge_index.shape[1])}

        self._for_each(slides, lambda s: s.id, "graph", lambda s: (sdir / f"{s.id}.pt").exists(), one, retry_failed)
        patients = [(p, ss) for p, ss in cohort.patients.items() if all(self._ready(s, "graph") for s in ss)]

        def merge(item):
            pid, ss = item
            ss = sorted(ss, key=lambda s: s.id)
            gs = [torch.load(sdir / f"{s.id}.pt", weights_only=False) for s in ss]
            G = patient_graph(pid, [s.id for s in ss], gs)
            torch.save(G, pdir / f"{_safe(pid)}.g")
            return {"slides": len(ss), "nodes": int(G.x.shape[0])}

        self._for_each(patients, lambda it: f"patient__{_safe(it[0])}", "patient_graph",
                       lambda it: (pdir / f"{_safe(it[0])}.g").exists(), merge, retry_failed, label="patient")

    def predict(self, cohort: Cohort, retry_failed: bool = True) -> pd.DataFrame:
        """Stage 5: model predictions per patient, plus per-slide overlays and heatmaps."""
        self._prepare(cohort)
        from .outputs import write_heatmap, write_overlay, write_thumbnail
        from .wsi import open_slide
        c = self.config
        model_cls = get_model(c.model)
        base = self.workspace / "predictions" / c.model
        paths = {s.id: s.path for s in cohort.slides}
        for d in ("patients", "local", "overlays", "heatmaps", "thumbnails"):
            (base / d).mkdir(parents=True, exist_ok=True)
        pdir = self.workspace / "graphs" / "patients"
        fdir = self.workspace / "features" / c.features.model
        patients = [(p, ss) for p, ss in cohort.patients.items()
                    if self._status(f"patient__{_safe(p)}").get("patient_graph", {}).get("status") == "done"]

        def make():
            from .models.remote import RemotePredictor, resolve_backend
            url = resolve_backend(c.backend, model_cls, c.weights_dir)
            if url:
                log.info("predict: sending patient graphs to %s (no images leave this machine)", url)
                return RemotePredictor(url, c.model, model_cls)
            return model_cls(self.device, c.weights_dir)

        with _model(make) as model:
            def one(item):
                pid, ss = item
                G = torch.load(pdir / f"{_safe(pid)}.g", weights_only=False)
                pred = model.predict(G)
                pred.patient = pid
                row = pred.row()
                (base / "patients" / f"{_safe(pid)}.json").write_text(json.dumps(row, default=float))
                ptr = G.slide_ptr.tolist()
                slide_idx = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
                if pred.local is not None:
                    np.savez_compressed(base / "local" / f"{_safe(pid)}.npz", local=pred.local, slide=slide_idx,
                                        slides=np.array(G.slideOrder))
                    for k, sid in enumerate(G.slideOrder):
                        z = np.load(fdir / f"{sid}.npz")
                        vals = pred.local[ptr[k]:ptr[k + 1]]
                        write_overlay(base / "overlays" / f"{sid}.db", z["bounds"] * float(z["baseline_scale"]), vals)
                        write_heatmap(base / "heatmaps" / f"{sid}.png", z["bounds"], vals)
                        try:
                            write_thumbnail(base / "thumbnails" / f"{sid}.jpg", open_slide(paths[sid], c.mpp),
                                            z["bounds"] * float(z["baseline_scale"]), vals)
                        except Exception as e:     # a thumbnail must never fail a prediction
                            log.warning("thumbnail for %s skipped: %s", sid, e)
                return {"risk": round(pred.risk, 5), "group": pred.group}

            self._for_each(patients, lambda it: f"patient__{_safe(it[0])}", "predict",
                           lambda it: (base / "patients" / f"{_safe(it[0])}.json").exists(), one, retry_failed, "patient")
        return self.predictions()

    # ── convenience ────────────────────────────────────────────────────────────────────────────────────────────────
    def run(self, cohort: Cohort, stages: Iterable[str] | None = None, retry_failed: bool = True) -> pd.DataFrame:
        """Run the given stages (default: all stages of the model) over the cohort and return its results table."""
        stages = list(stages or self.stages)
        bad = set(stages) - set(self.stages)
        if bad:
            raise ValueError(f"unknown stages {sorted(bad)} for {self.config.model}; choose from {self.stages}")
        self._prepare(cohort)
        if "predict" in stages:
            self._check_backend()
        log.info("HIMICO %s | %s | %s | workspace %s | device %s", __version__, self.config.model, cohort.summary(), self.workspace, self.device)
        for st in self.stages:
            if st in stages:
                getattr(self, st)(cohort, retry_failed=retry_failed)
        self.report(cohort)
        if "predict" in self.stages:
            log.info("view the results on the slides with: himico view %s", self.workspace)
        return self.predictions()

    def stain(self, cohort: Cohort, retry_failed: bool = True):
        """Virtual staining (e.g. HistoChrome): one pyramidal OME-TIFF per slide in virtual_stain/<model>/."""
        self._prepare(cohort)
        from .stages.virtual_stain import run_virtual_stain
        c = self.config
        out = self.workspace / "virtual_stain" / c.model
        out.mkdir(parents=True, exist_ok=True)
        with _model(lambda: get_model(c.model)(self.device, c.weights_dir)) as model:
            self._for_each(cohort.slides, lambda s: s.id, "stain", lambda s: (out / f"{s.id}.ome.tiff").exists(),
                           lambda s: run_virtual_stain(model, s, out, c.batch_size, c.workers, c.mpp), retry_failed)

    def _check_backend(self):
        """Warn before hours of preprocessing if the predict stage will have no API key."""
        from .auth import api_key, missing_key_message
        from .models.remote import resolve_backend
        url = resolve_backend(self.config.backend, get_model(self.config.model), self.config.weights_dir)
        if url and not api_key(url):
            log.warning("predict stage: %s Preprocessing runs anyway; add the key and re-run to finish.",
                        missing_key_message(url))

    def predictions(self) -> pd.DataFrame:
        """All patient predictions in the workspace (also written to predictions/<model>/predictions.csv); for virtual
        staining models, one row per slide with its output image."""
        if "stain" in self.stages:
            out = self.workspace / "virtual_stain" / self.config.model
            rows = [{"slide_id": p.name[:-len(".ome.tiff")], "image": str(p),
                     **self._status(p.name[:-len(".ome.tiff")]).get("stain", {}).get("info", {})}
                    for p in sorted(out.glob("*.ome.tiff"))] if out.exists() else []
            return pd.DataFrame(rows)
        base = self.workspace / "predictions" / self.config.model
        rows = [json.loads(p.read_text()) for p in sorted((base / "patients").glob("*.json"))] if base.exists() else []
        df = pd.DataFrame(rows)
        if len(df):
            df.to_csv(base / "predictions.csv", index=False)
        return df

    def report(self, cohort: Cohort | None = None) -> pd.DataFrame:
        """One row per slide: stage status, key numbers and errors. Also written to <workspace>/report.csv."""
        ws = self.workspace
        man = pd.read_csv(ws / "cohort.csv") if (ws / "cohort.csv").exists() else pd.DataFrame(columns=["slide_id", "patient_id"])
        if cohort is not None:
            man = man[man.slide_id.isin(cohort.slide_ids)]
        rows = []
        if "stain" in self.stages:
            for r in man.itertuples(index=False):
                s = self._status(r.slide_id).get("stain", {})
                rows.append({"slide_id": r.slide_id, "patient_id": r.patient_id, "stain": s.get("status", "pending"),
                             "tiles": s.get("info", {}).get("tiles"), "error": s.get("error", "")})
            df = pd.DataFrame(rows)
            if len(df):
                df.to_csv(ws / "report.csv", index=False)
                log.info("report: %d slides, %d stained, %d failed -> %s", len(df), (df.stain == "done").sum(),
                         (df.stain == "failed").sum(), ws / "report.csv")
            return df
        for r in man.itertuples(index=False):
            st, pst = self._status(r.slide_id), self._status(f"patient__{_safe(str(r.patient_id))}")
            row = {"slide_id": r.slide_id, "patient_id": r.patient_id}
            for stage in ("mask", "tissue", "features", "graph"):
                row[stage] = st.get(stage, {}).get("status", "pending")
            row["predict"] = pst.get("predict", {}).get("status", "pending")
            row["usable_tissue_mm2"] = st.get("mask", {}).get("info", {}).get("kept_mm2")
            row["patches"] = st.get("features", {}).get("info", {}).get("patches")
            row["risk"] = pst.get("predict", {}).get("info", {}).get("risk")
            row["risk_group"] = pst.get("predict", {}).get("info", {}).get("group")
            errs = [f"{k}: {v['error']}" for k, v in {**st, **pst}.items() if v.get("status") == "failed"]
            row["error"] = " | ".join(errs)
            rows.append(row)
        df = pd.DataFrame(rows)
        if len(df):
            df.to_csv(ws / "report.csv", index=False)
            done = (df.predict == "done").sum()
            failed = (df[["mask", "tissue", "features", "graph", "predict"]] == "failed").any(axis=1).sum()
            log.info("report: %d slides, %d predicted, %d with a failed stage -> %s", len(df), done, failed, ws / "report.csv")
        return df


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _model(factory):
    """Load a network for one stage and free it (and GPU memory) afterwards."""
    m = factory()
    try:
        yield m
    finally:
        del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(name))


def _environment() -> dict:
    import importlib.metadata as md
    env = {"himico": __version__, "python": platform.python_version(), "platform": platform.platform(),
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for pkg in ("torch", "timm", "tiatoolbox", "torch_geometric", "segmentation-models-pytorch", "openslide-python", "scipy",
                "numpy"):
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    return env
