# HIMICO SDK in Docker

One image contains the SDK and everything it needs (PyTorch, TIAToolbox, OpenSlide, the slide viewer). Slides are
mounted read-only and never leave the machine.

## Build

```bash
# NVIDIA GPU (x86-64)
docker build -f docker/Dockerfile -t himico:gpu .
# CPU only, also Apple silicon
docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t himico:cpu .
```

## Run a cohort

```bash
docker run --rm --gpus all \
  -v /path/to/slides:/data:ro \
  -v "$PWD/himico_runs":/output \
  -v himico-cache:/cache \
  -e HF_TOKEN -e HIMICO_API_KEY \
  himico:gpu run --slides /data --output /output --experiment my_cohort --patient-regex '^(TCGA-\w\w-\w{4})'
```

* `HF_TOKEN`: a Hugging Face token whose account has been approved for [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI).
  Preprocessing weights download once into the `himico-cache` volume; later runs reuse them.
* `HIMICO_API_KEY`: your HIMICO API key (free for academic research); the predict stage sends patch features and
  graph edges (no images) to the HIMICO server.
* Results appear in `./himico_runs/my_cohort/` (predictions CSV, heatmaps, thumbnails, overlays, report).
* Every `himico` command works the same way: `verify`, `report /output/my_cohort`, `bundle /output/my_cohort`, ...
* Files are written as user 1000; add `--user "$(id -u):$(id -g)"` to write as yourself.

Check the installation first: `docker run --rm --gpus all -v himico-cache:/cache -e HF_TOKEN -e HIMICO_API_KEY himico:gpu verify`.

## Offline / air-gapped sites

Build the image on a machine with internet, baking the preprocessing weights in with **your own** approved token
(the predict stage still needs the HIMICO server, or model weights you were given, mounted and passed with
`--weights-dir`):

```bash
export HF_TOKEN=...
docker build -f docker/Dockerfile --build-arg BAKE_WEIGHTS=1 --secret id=hf_token,env=HF_TOKEN -t himico:offline .
docker save himico:offline | gzip > himico_offline.tar.gz      # copy to the offline machine, then: docker load
```

The build runs `himico verify` so a broken image never leaves the build machine. The token is passed as a build
secret and is not stored in the image. Run offline images with `-e HF_HUB_OFFLINE=1` and without the `/cache` volume.
**UNI weights may not be redistributed: keep images built with `BAKE_WEIGHTS=1` inside your organisation.**

## Viewing results

The TIAToolbox viewer loads its image tiles from `127.0.0.1`, so inside a container it needs host networking and a
browser on the same Linux machine:

```bash
docker run --rm --network host -v /path/to/slides:/data:ro -v "$PWD/himico_runs":/output himico:cpu view /output/my_cohort
```

On macOS/Windows, or to view on another computer, install the SDK there (`pip install git+https://github.com/pkeller00/HIMICO-SDK`) and run
`himico view` on the workspace folder (slides must be at the same paths, or re-run with the slides mounted at `/data`).
