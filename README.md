# insight-crc

Run **INSIGHT** ([paper](https://arxiv.org/abs/2512.22262)) on whole-slide image cohorts: from raw slides to
patient-level survival risk, risk groups and patch-level risk maps, in one resumable command.

```
slides ──► 1 mask ──► 2 tissue ──► 3 features ──► 4 graphs ──► 5 predict
          GrandQC     CRC tissue     UNI          slide +       INSIGHT ensemble
          (or yours)  types          embeddings   patient       + overlays/heatmaps
```

Every stage runs over the whole cohort, caches its result per slide, and is skipped on the next run once done.
Failed slides are recorded and retried; one bad slide never stops a cohort.

## Install

```bash
pip install insight-crc            # Python ≥ 3.10; a CUDA GPU is recommended for cohorts
# or from source: pip install git+https://github.com/pkeller00/INSIGHT-SDK
insight doctor                     # checks packages, GPU, Hugging Face login and weights
```

Reading `.svs`, `.ndpi` or `.mrxs` slides needs OpenSlide (`pip install openslide-bin` or your OS package).

Hospitals and clusters can use the Docker image instead (GPU and CPU variants, optional offline build with weights):
see [docker/README.md](docker/README.md).

### Weights and access

| Component | Source | Access |
|---|---|---|
| INSIGHT (4 folds) + CRC tissue classifier | [pkeller01/INSIGHT](https://huggingface.co/pkeller01/INSIGHT) | request access (academic, non-commercial) |
| UNI feature extractor | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | request access from its authors |
| GrandQC tissue detection + artefact QC | [pkeller01/INSIGHT-preprocessing](https://huggingface.co/pkeller01/INSIGHT-preprocessing) | open (CC BY-NC-SA 4.0) |

Once approved, log in once with `hf auth login`; weights download on first use and are cached. On machines without
internet, copy them into a folder and pass `--weights-dir` (layout: `insight weights --weights-dir DIR`).
No access tokens are stored in this package.

## Quick start

```bash
insight run --slides /data/tcga_crc --experiment tcga --patient-regex '^(TCGA-\w\w-\w{4})'
```

Results land in `insight_runs/tcga/`:

```
predictions/insight-crc/predictions.csv     patient_id, score, risk, risk_group, fold scores, n_patches, n_slides
predictions/insight-crc/heatmaps/<slide>.png    predictions/insight-crc/thumbnails/<slide>.jpg  (risk over H&E)
predictions/insight-crc/overlays/<slide>.db  TIAToolbox overlay: open in the TIA viewer with the slide
report.csv                                   per-slide status, usable tissue area, patch counts, errors
```

Before a big run, check that everything is installed correctly:

```bash
insight verify        # runs the example TCGA graph and checks the known answer (risk 0.237, High), then loads
                      # GrandQC, the tissue classifier and UNI on a test tile
```

## Looking at the results

```bash
insight view insight_runs/tcga            # opens the TIAToolbox slide viewer in your browser
```

Pick a slide on the left, then **Add Overlay**:

| Overlay | Shows | Colour by |
|---|---|---|
| `<slide>_insight_risk.db` | INSIGHT local risk per patch (0 = lower, 1 = higher) | `risk` (blue to red) |
| `<slide>_tissue_types.db` | CRC tissue type per tile | `max_class` |
| `<slide>_qc_artefacts.png` | GrandQC artefacts (folds, pen, out of focus, ...) | fixed colours |

Hover a patch for its value; adjust overlay opacity with the slider. Patient-level results are in
`predictions/insight-crc/predictions.csv`, and `thumbnails/` has a quick risk-on-H&E image per slide. The viewer
reads slides in place and runs only on your computer; closing the browser tab stops it.

## Python API

```python
from insight import Cohort, Pipeline

cohort = Cohort.from_folder("/data/tcga_crc", patient=r"^(TCGA-\w\w-\w{4})")
# or: Cohort.from_csv("cohort.csv", slide_dir="/data/tcga_crc")   # slide_id/slide_path, patient_id
print(cohort.summary())

pipe = Pipeline(output="insight_runs", experiment="tcga", run=1, device="cuda", batch_size=128, workers=8)
preds = pipe.run(cohort)            # all stages; re-run to resume
pipe.report()                       # per-slide status DataFrame

# stages can also be run one by one
pipe.mask(cohort); pipe.tissue(cohort); pipe.features(cohort); pipe.graphs(cohort); pipe.predict(cohort)

# reopen a finished workspace with exactly the same settings
pipe = Pipeline.from_workspace("insight_runs/tcga/run_1")
```

In a script, put the calls under `if __name__ == "__main__":`. Tiles are read by worker processes, and on macOS and
Windows those re-import your script (or use `workers=0`).

## Common tasks

**Your own tissue masks.** Put `<slide_id>.png` (non-zero = usable tissue, covering the whole slide at any
resolution) in a folder and pass `--masks DIR` (or `Pipeline(masks=DIR)`); GrandQC is skipped.

**Several slides per patient.** Give a patient regex or a `patient_id` column; all of a patient's slides are merged
into one patient graph, and maps are still written per slide.

**Clusters.** Split a cohort across array jobs with `--shard i/n` (patients never split across shards); every job
writes into the same workspace. Pin a GPU with `--device cuda:1`.

**Slides without resolution metadata.** Pass `--mpp 0.25` as a fallback.

**One config file.** `insight init` writes `insight.yaml` with every option; run it with `insight run --config insight.yaml --slides DIR`.
The resolved config is saved to the workspace; a workspace refuses to mix results from different settings, so use a
new `--run` or `--variant` to compare settings side by side.

## Hosting the model for collaborators

`insight serve` puts the models behind a small REST API on a machine that holds the weights. Collaborators run the
heavy preprocessing (masks, tissue, UNI, graphs) on their own hardware and send only patient graphs, so the INSIGHT
weights never leave your server:

```bash
# on the GPU server
export INSIGHT_SERVER_TOKEN=$(openssl rand -hex 24)        # share with collaborators
insight serve --host 0.0.0.0 --port 8080 --preload insight-crc

# on the collaborator's machine
export INSIGHT_SERVER_TOKEN=...
insight verify --backend http://gpu-server:8080
insight run --slides /data/wsi --experiment cohort --backend http://gpu-server:8080
```

Endpoints: `GET /ping`, `GET /models`, `POST /predict/<model>` (body: `.g` bytes). Uploaded graphs are opened with a
restricted loader that cannot run code, validated, and never stored or logged. Without a token the server binds to
localhost only unless you choose otherwise; behind a public address always set a token (and ideally HTTPS via a
reverse proxy).

## Getting help

`insight bundle <workspace>` packs the config, environment, report, status files and logs into a `.tar.gz` with slide
IDs, patient IDs and file paths replaced by pseudonyms (no images, features or predictions). Look through it, then
attach it to an issue or email.

## Outputs per stage

```
<output>/<experiment>/run_<run>/<variant>/
  config.yaml  environment.json  cohort.csv  report.csv  logs/run_*.jsonl  status/<slide>.json
  masks/<slide>.png                    masks/qc/<slide>_classes.png, _overlay.jpg     (GrandQC classes 1-7)
  tissue/<slide>.npz                   tile bounds + 8-class probabilities (TUM NORM LYM MUS STR DEB MUC ADI)
  tissue/<slide>_mask.png              tissue types used for features (default TUM, STR, LYM)
  tissue/<slide>.db                    TIAToolbox overlay of tissue classes
  features/uni/<slide>.npz             patch bounds (0.5 µm/px) + 1024-d UNI embeddings
  graphs/slides/<slide>.pt             graphs/patients/<patient>.g   (PyTorch Geometric)
  predictions/<model>/...
```

Patient graphs keep the INSIGHT training format (`x`, `edge_index`, `coords`, `slideOrder`) and add `slide_ptr`
(node range of each slide), so they also run with `inference.py` from the
[GitHub repository](https://github.com/pkeller00/INSIGHT) and on the website's "Try INSIGHT" upload.

## Reproducibility

Checked against the original INSIGHT pipeline on a public TCGA patient (TCGA-4N-A93T):

* **Graphs:** identical edges, in the same order. Patches lie on a square grid, where Delaunay triangulation can pick
  either diagonal of a square and the choice depends on floating-point rounding. INSIGHT's training graphs were built
  with Qhull 2019.1 compiled without fused multiply-add, so the SDK ships its own copy of that Qhull
  (`src/insight/_qhull`, Qhull licence) compiled with `-ffp-contract=off`, rather than relying on SciPy.
* **Features:** UNI features at the original patch positions match with cosine similarity ≥ 0.99998.
* **Predictions:** patient risk 0.237039 vs 0.237084 and per-patch Spearman 1.000 (remaining differences are
  floating-point rounding across hardware).

From raw slides, where masks are recomputed, IMMUcan IMU004 gives risk 0.475 vs 0.480 in the original run.
Re-running the SDK on the same machine is deterministic.

## Adding a model

Models are looked up by name, so a new model is one class:

```python
from insight import register_model
from insight.models import Predictor, Prediction

@register_model("my-model")
class MyModel(Predictor):
    features = "uni"          # feature extractor it was trained on
    dthresh = 4000            # graph edge distance used in training
    description = "..."

    def __init__(self, device, weights_dir=None):
        super().__init__(device, weights_dir)
        ...                    # load weights (see insight.weights.resolve)

    def predict(self, graph) -> Prediction:
        ...                    # return Prediction(patient, score, risk, group, folds, local)
```

Then `Pipeline(model="my-model")` or `insight run --model my-model`. Packages can also expose models through the
`insight.models` entry point. New feature extractors register with `@register_features("name")`.

## Citation and licence

The SDK and INSIGHT weights are released under [CC BY-NC-ND 4.0](LICENSE): academic, non-commercial research use only; not a medical device. Please cite:

```bibtex
@article{keller2025insight,
  title   = {INSIGHT: Spatially resolved survival modelling from routine histology crosslinked with molecular
             profiling reveals prognostic epithelial-immune axes in stage II/III colorectal cancer},
  author  = {Keller, Piotr and others and Minhas, Fayyaz},
  journal = {arXiv preprint arXiv:2512.22262},
  year    = {2025}
}
```

The pipeline builds on [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox),
[GrandQC](https://github.com/cpath-ukk/grandqc) (CC BY-NC-SA 4.0) and [UNI](https://huggingface.co/MahmoodLab/UNI);
please also cite them and follow their licences.
