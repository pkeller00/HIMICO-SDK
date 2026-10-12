"""`insight` command line.

    insight init                      write a starter insight.yaml
    insight run --slides /data/wsi --experiment tcga [--patient-regex RE] [--masks DIR] [--shard 0/4]
    insight run --config insight.yaml --slides /data/wsi
    insight report runs/tcga/run_1
    insight models | insight weights [--download] | insight doctor
    insight verify [--backend URL]     known-answer check before a big run
    insight serve [--host 0.0.0.0]     host models for collaborators (INSIGHT_SERVER_TOKEN)
    insight bundle WORKSPACE           privacy-safe support file
    insight view WORKSPACE             browse slides with risk / tissue / QC overlays (TIAToolbox viewer)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cohort(a):
    from .cohort import Cohort

    if a.csv:
        c = Cohort.from_csv(a.csv, slide_dir=a.slides)
    elif a.slides:
        c = Cohort.from_folder(a.slides, patient=a.patient_regex, recursive=not a.no_recursive)
    else:
        sys.exit("give --slides DIR or --csv cohort.csv")
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        c = c.shard(i, n)
    if not len(c):
        sys.exit("no slides found")
    return c


def cmd_init(a):
    from .config import template_yaml

    p = Path(a.path)
    if p.exists() and not a.force:
        sys.exit(f"{p} exists (use --force to overwrite)")
    p.write_text(template_yaml())
    print(f"wrote {p}. Edit it, then: insight run --config {p} --slides /path/to/slides")


def cmd_run(a):
    from .config import PipelineConfig
    from .pipeline import STAGES, Pipeline

    cfg = PipelineConfig.from_yaml(a.config) if a.config else PipelineConfig()
    over = {k: v for k, v in dict(output=a.output, experiment=a.experiment, run=a.run, variant=a.variant, model=a.model,
                                  device=a.device, batch_size=a.batch_size, workers=a.workers, mpp=a.mpp,
                                  weights_dir=a.weights_dir, backend=a.backend).items() if v is not None}
    if a.masks:
        over["masks"] = a.masks
    pipe = Pipeline(cfg, **over)
    cohort = _cohort(a)
    stages = a.stages.split(",") if a.stages else STAGES
    preds = pipe.run(cohort, stages=stages, retry_failed=not a.skip_failed)
    if len(preds):
        print(preds.to_string(index=False, max_rows=30))
    print(f"\nworkspace: {pipe.workspace}")


def cmd_report(a):
    from .pipeline import Pipeline

    pipe = Pipeline.from_workspace(a.workspace)
    from .log import setup
    setup(None)
    df = pipe.report()
    if not len(df):
        sys.exit("no slides recorded in this workspace yet")
    cols = ["slide_id", "patient_id", "mask", "tissue", "features", "graph", "predict", "risk", "risk_group"]
    print(df[cols].to_string(index=False, max_rows=a.rows))
    failed = df[df.error != ""]
    if len(failed):
        print("\nfailures:")
        for r in failed.itertuples():
            print(f"  {r.slide_id}: {r.error}")


def cmd_models(a):
    from .registry import feature_extractors, models

    print("prediction models:")
    for name, cls in models().items():
        print(f"  {name:<16} {cls.description}  [features: {cls.features}]  {cls.paper}")
    print("feature extractors:")
    for name, cls in feature_extractors().items():
        print(f"  {name:<16} dim {cls.dim}")


def cmd_weights(a):
    from . import weights

    rows = weights.status(a.weights_dir)
    for key, st in rows:
        print(f"  {st:<8} {key}")
    if a.download:
        print("\ndownloading missing files ...")
        for key, st in rows:
            if st == "missing":
                try:
                    print("  ok      ", key, "->", weights.resolve(key, a.weights_dir))
                except weights.WeightsUnavailable as e:
                    print("  FAILED  ", e)


def cmd_doctor(a):
    import importlib.metadata as md
    import platform

    ok = True
    print(f"python {platform.python_version()} on {platform.platform()}")
    for pkg in ("torch", "torchvision", "timm", "tiatoolbox", "torch_geometric", "segmentation-models-pytorch",
                "safetensors", "huggingface_hub", "openslide-python"):
        try:
            print(f"  {pkg:<30} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:<30} MISSING")
            ok = False
    import torch
    from .wsi import device_for
    print(f"device (auto): {device_for('auto')}" + (f"  [{torch.cuda.get_device_name(0)}]" if torch.cuda.is_available() else ""))
    try:
        from huggingface_hub import whoami
        print(f"hugging face: logged in as {whoami()['name']}")
    except Exception:
        print("hugging face: not logged in (needed for gated INSIGHT and UNI weights: `hf auth login`)")
    missing = [k for k, s in __import__("insight.weights", fromlist=["status"]).status(a.weights_dir) if s == "missing"]
    print(f"weights: {'all available' if not missing else str(len(missing)) + ' not downloaded yet (see `insight weights`)'}")
    sys.exit(0 if ok else 1)


def cmd_verify(a):
    from .support import verify

    res = verify(a.device or "auto", a.weights_dir, a.backend, a.quick)
    for name, ok, detail in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    bad = [r for r in res if not r[1]]
    print(f"\n{len(res) - len(bad)}/{len(res)} checks passed")
    sys.exit(1 if bad else 0)


def cmd_serve(a):
    from .log import setup
    from .server import serve

    setup(None)
    serve(a.host, a.port, a.device or "auto", a.weights_dir, preload=a.preload.split(",") if a.preload else None)


def cmd_bundle(a):
    from .support import bundle

    out = bundle(a.workspace, a.out)
    print(f"wrote {out}\nSlide IDs, patient IDs and paths are pseudonymised. Please look through it before sharing.")


def cmd_view(a):
    from .view import prepare, view

    if a.prepare_only:
        print(f"viewer files ready in {prepare(a.workspace)}")
        return
    sys.exit(view(a.workspace, a.port, a.noshow))


def main(argv=None):
    p = argparse.ArgumentParser(prog="insight", description="Run INSIGHT on whole-slide image cohorts.")
    p.add_argument("--version", action="version", version=__import__("insight").__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write a starter config file")
    s.add_argument("path", nargs="?", default="insight.yaml")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("run", help="run the pipeline over a cohort (resumable)")
    s.add_argument("--config", help="YAML config (see `insight init`)")
    s.add_argument("--slides", help="folder of whole-slide images")
    s.add_argument("--csv", help="cohort CSV: slide_path or slide_id, optional patient_id")
    s.add_argument("--patient-regex", help=r"regex giving the patient id from the slide id, e.g. '^(TCGA-\w\w-\w{4})'")
    s.add_argument("--no-recursive", action="store_true")
    s.add_argument("--masks", help="folder of your own <slide_id>.png masks (skips GrandQC)")
    s.add_argument("--output")
    s.add_argument("--experiment")
    s.add_argument("--run", type=int)
    s.add_argument("--variant")
    s.add_argument("--model")
    s.add_argument("--device")
    s.add_argument("--batch-size", type=int)
    s.add_argument("--workers", type=int)
    s.add_argument("--mpp", type=float, help="fallback microns-per-pixel for slides without it")
    s.add_argument("--weights-dir")
    s.add_argument("--backend", help="run the model on an `insight serve` server, e.g. http://gpu-box:8080")
    s.add_argument("--stages", help="comma list of mask,tissue,features,graphs,predict (default: all)")
    s.add_argument("--shard", help="i/n: process every n-th patient starting at i (for array jobs)")
    s.add_argument("--skip-failed", action="store_true", help="do not retry slides that failed before")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="per-slide status of a workspace")
    s.add_argument("workspace")
    s.add_argument("--rows", type=int, default=50)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("models", help="list prediction models and feature extractors")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("weights", help="show (and optionally download) model weights")
    s.add_argument("--weights-dir")
    s.add_argument("--download", action="store_true")
    s.set_defaults(fn=cmd_weights)

    s = sub.add_parser("doctor", help="check the environment")
    s.add_argument("--weights-dir")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("verify", help="known-answer check of models and weights before a big run")
    s.add_argument("--device")
    s.add_argument("--weights-dir")
    s.add_argument("--backend", help="check an `insight serve` server instead of local weights")
    s.add_argument("--quick", action="store_true", help="only the INSIGHT model check")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("serve", help="host models behind a REST API (set INSIGHT_SERVER_TOKEN to require a token)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--device")
    s.add_argument("--weights-dir")
    s.add_argument("--preload", help="comma list of models to load at start-up (default: on first request)")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("bundle", help="privacy-safe support file (config, environment, logs; IDs pseudonymised)")
    s.add_argument("workspace")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_bundle)

    s = sub.add_parser("view", help="open a workspace in the TIAToolbox slide viewer (slides + risk/tissue/QC overlays)")
    s.add_argument("workspace")
    s.add_argument("--port", type=int, default=5006)
    s.add_argument("--noshow", action="store_true", help="do not open a browser")
    s.add_argument("--prepare-only", action="store_true", help="only build <workspace>/viewer/")
    s.set_defaults(fn=cmd_view)

    a = p.parse_args(argv)
    from .pipeline import ConfigMismatch
    from .weights import WeightsUnavailable
    try:
        a.fn(a)
    except (ConfigMismatch, WeightsUnavailable, FileNotFoundError, ValueError) as e:
        sys.exit(f"insight: {e}")
    except KeyboardInterrupt:
        sys.exit("interrupted; re-run the same command to resume")


if __name__ == "__main__":
    main()
