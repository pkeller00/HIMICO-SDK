# INSIGHT in Docker

One image contains the SDK and everything it needs (PyTorch, TIAToolbox, OpenSlide, the slide viewer). Slides are
mounted read-only and never leave the machine.

## Build

```bash
# NVIDIA GPU (x86-64)
docker build -f docker/Dockerfile -t insight:gpu .
# CPU only, also Apple silicon
docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t insight:cpu .
```

## Run a cohort

```bash
docker run --rm --gpus all \
  -v /path/to/slides:/data:ro \
  -v "$PWD/insight_runs":/output \
  -v insight-cache:/cache \
  -e HF_TOKEN \
  insight:gpu run --slides /data --output /output --experiment my_cohort --patient-regex '^(TCGA-\w\w-\w{4})'
```

* `HF_TOKEN`: a Hugging Face token whose account has been approved for
  [pkeller01/INSIGHT](https://huggingface.co/pkeller01/INSIGHT) and [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI).
  Weights download once into the `insight-cache` volume; later runs reuse them.
* Results appear in `./insight_runs/my_cohort/` (predictions CSV, heatmaps, thumbnails, overlays, report).
* Every `insight` command works the same way: `verify`, `report /output/my_cohort`, `bundle /output/my_cohort`, ...
* Files are written as user 1000; add `--user "$(id -u):$(id -g)"` to write as yourself.

Check the installation first: `docker run --rm --gpus all -v insight-cache:/cache -e HF_TOKEN insight:gpu verify`.

## Offline / air-gapped sites

Build the image on a machine with internet, baking the weights in with **your own** approved token:

```bash
export HF_TOKEN=...
docker build -f docker/Dockerfile --build-arg BAKE_WEIGHTS=1 --secret id=hf_token,env=HF_TOKEN -t insight:offline .
docker save insight:offline | gzip > insight_offline.tar.gz      # copy to the offline machine, then: docker load
```

The build runs `insight verify` so a broken image never leaves the build machine. The token is passed as a build
secret and is not stored in the image. Run offline images with `-e HF_HUB_OFFLINE=1` and without the `/cache` volume.
**INSIGHT and UNI weights may not be redistributed: keep images built with `BAKE_WEIGHTS=1` inside your organisation.**

## Host the model for collaborators

`docker-compose.yml` runs `insight serve` on a GPU with a token:

```bash
export INSIGHT_SERVER_TOKEN=$(openssl rand -hex 24) HF_TOKEN=...
docker compose -f docker/docker-compose.yml up -d
```

Collaborators then run the pipeline on their own data with `--backend http://<host>:8080` and the same
`INSIGHT_SERVER_TOKEN`; only patient graphs are sent. Put a TLS reverse proxy in front if the port is reachable beyond
your network.

## Viewing results

The TIAToolbox viewer loads its image tiles from `127.0.0.1`, so inside a container it needs host networking and a
browser on the same Linux machine:

```bash
docker run --rm --network host -v /path/to/slides:/data:ro -v "$PWD/insight_runs":/output insight:cpu view /output/my_cohort
```

On macOS/Windows, or to view on another computer, install the SDK there (`pip install insight-crc`) and run
`insight view` on the workspace folder (slides must be at the same paths, or re-run with the slides mounted at `/data`).
