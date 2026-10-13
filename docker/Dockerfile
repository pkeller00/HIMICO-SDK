# HIMICO SDK container.
#
# GPU (default, x86-64 + NVIDIA):   docker build -f docker/Dockerfile -t himico:gpu .
# CPU / Apple silicon:              docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t himico:cpu .
#
# Offline sites can bake the preprocessing weights in at build time with their OWN approved Hugging Face token (UNI is
# gated and must not be redistributed, so an image built this way must stay inside your organisation):
#   docker build -f docker/Dockerfile --build-arg BAKE_WEIGHTS=1 --secret id=hf_token,env=HF_TOKEN -t himico:offline .
#
# Run (slides read-only, results to ./himico_runs):
#   docker run --rm --gpus all -v /path/to/slides:/data:ro -v $PWD/himico_runs:/output -v himico-cache:/cache \
#       -e HF_TOKEN -e HIMICO_API_KEY himico:gpu run --slides /data --output /output --experiment my_cohort
# See docker/README.md for serving, offline use and the viewer.

ARG VARIANT=gpu

# ── base images ──────────────────────────────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base-gpu
ENV TORCH_INDEX=""

FROM python:3.12-slim AS base-cpu
ENV TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"

FROM base-${VARIANT} AS final
ARG BAKE_WEIGHTS=0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    HIMICO_IN_CONTAINER=1

# OpenCV runtime libraries (OpenSlide itself comes from the openslide-bin wheel)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU variant: install CPU wheels of torch first so the SDK does not pull CUDA builds
RUN if [ -n "$TORCH_INDEX" ]; then pip install $TORCH_INDEX "torch>=2.5" "torchvision>=0.20"; fi

WORKDIR /opt/himico
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# a C compiler is needed only to build the vendored Qhull extension; removed afterwards
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && pip install ".[viewer]" "openslide-bin" && himico --version \
    && python -c "import himico._qhull_legacy" \
    && apt-get purge -y gcc libc6-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Non-root user; /data (slides, mount read-only), /output (workspaces), /cache (downloaded weights)
RUN useradd --create-home --uid 1000 himico && mkdir -p /data /output /cache/hf && chown -R himico /output /cache
USER himico

# Optional: bake weights into the image (site-internal builds only; see above)
RUN --mount=type=secret,id=hf_token,uid=1000 \
    if [ "$BAKE_WEIGHTS" = "1" ]; then \
        export HF_TOKEN="$(cat /run/secrets/hf_token)" && \
        himico weights --download && himico verify --device cpu --preprocessing-only; \
    fi

VOLUME ["/output", "/cache"]
EXPOSE 8080
ENTRYPOINT ["himico"]
CMD ["--help"]
