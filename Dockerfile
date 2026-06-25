# BeeWings REST API — CPU image.
#
# Build:  docker build -t beewings-api .
# Run:    docker run -p 8000:8000 beewings-api
#
# Note: checkpoints/*.pt in the repo are symlinks into runs/. Docker cannot
# follow them into the image, so we COPY the real weight files explicitly.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BEEWINGS_CHECKPOINTS=/app/checkpoints \
    BEEWINGS_API_HOST=0.0.0.0 \
    BEEWINGS_API_PORT=8000

# OpenCV runtime libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of torch first so it lands in its own cached layer and
# pip does not pull the much larger CUDA wheel to satisfy torch>=2.1.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Project metadata + source, then install with the API extra.
# A C/C++ toolchain is needed only to build `stringzilla` (pulled in transitively
# by albumentations, no prebuilt wheel for this platform). Install it, build, then
# purge it in the same layer so it does not bloat the final image.
COPY pyproject.toml README.md ./
COPY beewings ./beewings
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install ".[api]" \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Real model weights (resolve the repo symlinks to concrete filenames).
COPY runs/alpatov12_v1/best.pt /app/checkpoints/alpatov12.pt
COPY runs/unet19_v1/best.pt /app/checkpoints/tofilski19.pt

EXPOSE 8000
CMD ["beewings-api"]
