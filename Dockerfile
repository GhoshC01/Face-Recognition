# syntax=docker/dockerfile:1
#
# CPU-only production image for the Face Verification API.
# Two stages: a "builder" that resolves Python dependencies into a venv, and
# a slim "runtime" that copies only that venv + application code -- keeps
# the final image free of pip's build cache and any compiler toolchain.

########################################
# Stage 1: builder
########################################
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# onnxruntime, faiss-cpu, and opencv-python-headless all ship prebuilt
# manylinux wheels for this base image's platform/Python version, so no
# compiler toolchain is required here -- a plain `pip install` is enough.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

########################################
# Stage 2: runtime
########################################
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Face Verification API" \
      org.opencontainers.image.description="CPU-only face detection/recognition/verification service (SCRFD + MobileFaceNet + FAISS)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app

# libgl1 + libglib2.0-0: even the "headless" OpenCV wheel's compiled .so
# still resolves these shared library symbols at import time on minimal
# Debian images (a long-documented quirk of opencv-python-headless, not a
# GUI dependency) -- both are small and pull in no GUI stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /usr/sbin/nologin --create-home appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR ${APP_HOME}
COPY app ./app
# Operational tooling (accuracy benchmarking, stale-enrollment purging) is
# included so it can be run inside the running container -- e.g.
# `docker exec <container> python scripts/purge_stale_enrollments.py ...` --
# against the same live storage volume, with the same settings/env vars.
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Model binaries (models/*.onnx) and persisted index/metadata (storage/) are
# deliberately NOT baked into this image -- see docs/deployment.md "Models"
# and "Persistence". These directories exist here only so a read-only bind
# mount (models/) or a named volume (storage/) has a valid, correctly-owned
# mount point to attach to at `docker run` time.
RUN mkdir -p models storage/faiss storage/metadata \
    && chown -R appuser:appuser ${APP_HOME}

USER appuser

EXPOSE 8000

# Uses /health/ready (not /health/live): only reports healthy once both
# ONNX models have actually loaded and the vector store is reachable.
# start-period gives model loading time to complete before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
