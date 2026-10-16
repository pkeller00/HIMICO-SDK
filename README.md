# HIMICO SDK

Run the HIMICO group's computational pathology models on your own whole-slide images, from raw slides to patient-level
predictions and patch-level maps, in one resumable command.

| Model | Task | Paper |
|---|---|---|
| `insight-crc` | **INSIGHT**: survival risk, risk groups and patch-level risk maps in stage II/III colorectal cancer | [arXiv:2512.22262](https://arxiv.org/abs/2512.22262) |
| `histochrome` | **HistoChrome**: virtual multiplex IHC from H&E ([details](#histochrome-virtual-multiplex-ihc)) | [HistoChromeCode](https://github.com/measty/HistoChromeCode) |

**How it works.** Slide processing (tissue masking, feature extraction, graph building) runs on your own machine. For
INSIGHT, only the patient graph (patch features and edges: no images, coordinates or IDs) is sent to the HIMICO server
at the University of Warwick, which holds the model and returns the prediction. Graphs are not stored or logged.

## 1. Install, log in, verify

```bash
pip install git+https://github.com/pkeller00/HIMICO-SDK   # Python 3.10 or newer; a CUDA GPU is recommended
pip install openslide-bin                                 # to read .svs / .ndpi / .mrxs slides

himico login            # paste your HIMICO API key (request one: Piotr.Keller@warwick.ac.uk, free for academic research)
hf auth login           # Hugging Face account approved for the UNI feature extractor (https://huggingface.co/MahmoodLab/UNI)

himico verify --model insight-crc --quick   # checks your key and the server with a known example (expect risk 0.237, High)
```

`himico doctor` checks your packages, GPU, logins and key if something does not work.

## 2. Run INSIGHT on your cohort

Put your slides in one folder and point the SDK at it:

```bash
himico verify --model insight-crc                # full check, once: also loads the preprocessing models on a test tile

himico run --model insight-crc --slides /path/to/slides --experiment my_cohort
```

If a patient has several slides, tell it how to get the patient ID from the file name (all of a patient's slides are
merged into one graph), for example `--patient-regex '^(TCGA-\w\w-\w{4})'`.

The run is resumable: if it stops, run the same command again. Results are in `himico_runs/my_cohort/`:

```
predictions/insight-crc/predictions.csv    one row per patient: risk, risk_group (High / Low), fold scores
predictions/insight-crc/heatmaps/          patch-level risk drawn over each slide
report.csv                                 per-slide status and any errors
```

Browse the maps with `himico view himico_runs/my_cohort`.

### Already have patient graphs?

If you already built INSIGHT patient graphs (`.g` files), skip slide processing and score them directly:

```python
from pathlib import Path
import pandas as pd
from himico.auth import server_url
from himico.registry import get_model
from himico.safe_graph import load_graph
from himico.models.remote import RemotePredictor
import himico.models  # registers insight-crc

pred = RemotePredictor(server_url(), "insight-crc", get_model("insight-crc"))
rows = []
for f in sorted(Path("/path/to/graphs").glob("*.g")):
    r = pred.predict(load_graph(f.read_bytes()))
    rows.append({"patient": f.stem, "risk": r.risk, "risk_group": r.group})
pd.DataFrame(rows).to_csv("predictions.csv", index=False)
```

---

# More

## Pipeline

```
slides ──► 1 mask ──► 2 tissue ──► 3 features ──► 4 graphs ──► 5 predict
          GrandQC     CRC tissue     UNI          slide +       model on the HIMICO server
          (or yours)  types          embeddings   patient       + overlays/heatmaps
```

Stages 1-4 always run on your own hardware. Every stage runs over the whole cohort, caches its result per slide, and
is skipped on the next run once done. Failed slides are recorded and retried; one bad slide never stops a cohort.

## API key

```bash
himico login      # prompts for the key; stored in ~/.config/himico/credentials.json (mode 600)
himico account    # name, tier and models of your key
# clusters: export HIMICO_API_KEY=... instead
```

Free for non-commercial academic research; commercial licences on request (Piotr.Keller@warwick.ac.uk). The pipeline
warns at the start of a run if no key is set, so preprocessing a cohort never ends in a surprise.

## Preprocessing weights

| Component | Source | Access |
|---|---|---|
| GrandQC tissue detection + artefact QC | [pkeller01/INSIGHT-preprocessing](https://huggingface.co/pkeller01/INSIGHT-preprocessing) | open (CC BY-NC-SA 4.0) |
| CRC tissue classifier | [pkeller01/INSIGHT-preprocessing](https://huggingface.co/pkeller01/INSIGHT-preprocessing) | open (CC BY-NC-ND 4.0) |
| UNI feature extractor | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | request access from its authors, then `hf auth login` |

These download on first use and are cached. On machines without internet, copy them into a folder and pass
`--weights-dir` (layout: `himico weights --weights-dir DIR`). No access tokens are stored in this package.

## Looking at the results

```bash
himico view himico_runs/tcga            # opens the TIAToolbox slide viewer in your browser
```

Pick a slide on the left, then **Add Overlay**:

| Overlay | Shows | Colour by |
|---|---|---|
| `<slide>_insight_risk.db` | INSIGHT local risk per patch (0 = lower, 1 = higher) | `insight_risk` (blue to red) |
| `<slide>_tissue_types.db` | CRC tissue type per tile | `max_class` |
| `<slide>_qc_artefacts.db` | GrandQC artefacts (folds, pen, out of focus, ...) | `artefact` |

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
(node range of each slide).

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
