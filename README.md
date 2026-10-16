# HIMICO SDK

One SDK for the HIMICO group's computational pathology models: from raw whole-slide images to patient-level
predictions and patch-level maps, in one resumable command. Models available:

| Model | Task | Paper |
|---|---|---|
| `insight-crc` | **INSIGHT**: overall-survival risk, risk groups and patch-level risk maps in stage II/III colorectal cancer | [arXiv:2512.22262](https://arxiv.org/abs/2512.22262) |
| `histochrome` | **HistoChrome**: virtual multiplex IHC from H&E, as a whole-slide image ([details](#histochrome-virtual-multiplex-ihc)) | [HistoChromeCode](https://github.com/measty/HistoChromeCode) |
| SPARROW, ... | coming soon | |

Every command that runs a model names it with `--model` (or `model:` in a config file); `himico models` lists them.

```
slides ──► 1 mask ──► 2 tissue ──► 3 features ──► 4 graphs ──► 5 predict
          GrandQC     CRC tissue     UNI          slide +       model on the HIMICO server
          (or yours)  types          embeddings   patient       + overlays/heatmaps
```

Stages 1-4 always run on your own hardware. Every stage runs over the whole cohort, caches its result per slide, and
is skipped on the next run once done. Failed slides are recorded and retried; one bad slide never stops a cohort.

## Install

```bash
pip install git+https://github.com/pkeller00/HIMICO-SDK    # Python ≥ 3.10; a CUDA GPU is recommended for cohorts
himico doctor                     # checks packages, GPU, Hugging Face login, weights and your API key
```

Reading `.svs`, `.ndpi` or `.mrxs` slides needs OpenSlide (`pip install openslide-bin` or your OS package).

### API key

Model weights (INSIGHT and future HIMICO models) are not distributed. Predictions run on the HIMICO server at the
University of Warwick: for each patient the SDK sends only the UNI patch features and the graph edges (no images,
no patch coordinates, no slide or patient IDs) and receives the prediction and patch-level map. Graphs are not
stored or logged.

1. Request a key: free for non-commercial academic research; commercial licences on request
   (Piotr.Keller@warwick.ac.uk).
2. Save it once:

```bash
himico login                     # prompts for the key; stored in ~/.config/himico/credentials.json (mode 600)
himico account                   # name, tier and models of your key
# clusters / containers: export HIMICO_API_KEY=... instead
```

The model runs on the HIMICO server by default. `backend` can also be the URL of a `himico serve` server. The pipeline
warns at the start of a run if no key is set, so preprocessing a cohort never ends in a surprise.

### Preprocessing weights

| Component | Source | Access |
|---|---|---|
| GrandQC tissue detection + artefact QC | [pkeller01/INSIGHT-preprocessing](https://huggingface.co/pkeller01/INSIGHT-preprocessing) | open (CC BY-NC-SA 4.0) |
| CRC tissue classifier | [pkeller01/INSIGHT-preprocessing](https://huggingface.co/pkeller01/INSIGHT-preprocessing) | open (CC BY-NC-ND 4.0) |
| UNI feature extractor | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | request access from its authors, then `hf auth login` |

These download on first use and are cached. On machines without internet, copy them into a folder and pass
`--weights-dir` (layout: `himico weights --weights-dir DIR`). No access tokens are stored in this package.

## Quick start

```bash
himico run --model insight-crc --slides /data/tcga_crc --experiment tcga --patient-regex '^(TCGA-\w\w-\w{4})'
```

Results land in `himico_runs/tcga/`:

```
predictions/insight-crc/predictions.csv     patient_id, score, risk, risk_group, fold scores, n_patches, n_slides
predictions/insight-crc/heatmaps/<slide>.png    predictions/insight-crc/thumbnails/<slide>.jpg  (risk over H&E)
predictions/insight-crc/overlays/<slide>.db  TIAToolbox overlay: open in the TIA viewer with the slide
report.csv                                   per-slide status, usable tissue area, patch counts, errors
```

Before a big run, check that everything is installed correctly:

```bash
himico verify --model insight-crc   # sends the example TCGA graph to the server and checks the known answer (risk 0.237,
                      # High), then loads GrandQC, the tissue classifier and UNI on a test tile
```

## Looking at the results

```bash
himico view himico_runs/tcga            # opens the TIAToolbox slide viewer in your browser
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
from himico import Cohort, Pipeline

cohort = Cohort.from_folder("/data/tcga_crc", patient=r"^(TCGA-\w\w-\w{4})")
# or: Cohort.from_csv("cohort.csv", slide_dir="/data/tcga_crc")   # slide_id/slide_path, patient_id
print(cohort.summary())

pipe = Pipeline(model="insight-crc", output="himico_runs", experiment="tcga", run=1, device="cuda", batch_size=128, workers=8)
preds = pipe.run(cohort)            # all stages; re-run to resume
pipe.report()                       # per-slide status DataFrame

# stages can also be run one by one
pipe.mask(cohort); pipe.tissue(cohort); pipe.features(cohort); pipe.graphs(cohort); pipe.predict(cohort)

# reopen a finished workspace with exactly the same settings
pipe = Pipeline.from_workspace("himico_runs/tcga/run_1")
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

**One config file.** `himico init` writes `himico.yaml` with every option; run it with `himico run --config himico.yaml --slides DIR` (the file names the model).
The resolved config is saved to the workspace; a workspace refuses to mix results from different settings, so use a
new `--run` or `--variant` to compare settings side by side.

## HistoChrome: virtual multiplex IHC

HistoChrome (HIMICO, [HistoChromeCode](https://github.com/measty/HistoChromeCode)) turns a routine H&E slide into a
virtual 5-marker multiplex immunohistochemistry (mIHC) image. It runs entirely on your own hardware; no API key is
involved. **The weights are not public yet**: ask the HIMICO team, then put the file at
`<weights_dir>/histochrome/histochrome_he2mihc.safetensors` (or use a Hugging Face account that has been given access).

```bash
himico verify --model histochrome --weights-dir /path/to/weights
himico run --model histochrome --slides /data/he_slides --experiment mihc --weights-dir /path/to/weights
```

```
<output>/<experiment>/virtual_stain/histochrome/<slide>.ome.tiff    pyramidal OME-TIFF (RGB, JPEG q87) at 0.5 um/px
<output>/<experiment>/virtual_stain/histochrome/<slide>_thumb.jpg   quick look
report.csv                                                       per-slide status, tiles, errors
```

Each slide is read in 832 px tiles at 0.5 um/px with 96 px overlap; only tiles with at least 20% tissue (Otsu mask)
are processed, overlaps are blended with a Hann window and the background is white, exactly as in HistoChromeCode's
`infer.py` (the SDK's network gives identical output on real H&E tiles). Tiles are stitched in a rolling band of
rows, so no large temporary canvas is needed (about 0.7 GB of memory for a 50,000 px wide slide). A GPU is strongly
recommended: an 832 px tile takes about 3 s on a laptop GPU and about 30 s on a laptop CPU. Open the result in QuPath
or any OME-TIFF viewer.

## Getting help

`himico bundle <workspace>` packs the config, environment, report, status files and logs into a `.tar.gz` with slide
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
(node range of each slide), so they also run on the INSIGHT website's "Try INSIGHT" upload.

## Adding a model

Models are looked up by name, so a new model is one class:

```python
from himico import register_model
from himico.models import Predictor, Prediction

@register_model("my-model")
class MyModel(Predictor):
    features = "uni"          # feature extractor it was trained on
    dthresh = 4000            # graph edge distance used in training
    description = "..."

    def __init__(self, device, weights_dir=None):
        super().__init__(device, weights_dir)
        ...                    # load weights (see himico.weights.resolve)

    def predict(self, graph) -> Prediction:
        ...                    # return Prediction(patient, score, risk, group, folds, local)
```

Then `Pipeline(model="my-model")` or `himico run --model my-model`. Packages can also expose models through the
`himico.models` entry point. New feature extractors register with `@register_features("name")`.

## Citation and licence

The SDK is released under [CC BY-NC-ND 4.0](LICENSE): academic, non-commercial research use only; not a medical
device. Commercial use of the SDK or the hosted models needs a licence (Piotr.Keller@warwick.ac.uk). If you use
INSIGHT, please cite:

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
