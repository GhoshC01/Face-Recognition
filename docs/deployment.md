# Deployment Guide (Docker, CPU-only production)

This covers containerizing and running the Face Verification API in production: the Docker image
itself, model/index persistence, ASGI startup configuration, sizing, backup/recovery, model
versioning, and local testing. It assumes `SECURITY.md` (auth, CORS, rate limiting, upload
validation, TLS) is already read — this document is about *running* the container, not hardening
the application logic.

**A note on verification**: the Dockerfile, compose file, and entrypoint script in this repo were
written and reviewed carefully, but this development environment does not have Docker available to
actually execute `docker build`/`docker run` against them. Treat the build as unverified until you
run it yourself locally (see "Local Docker testing" below) — please report back if anything doesn't
build or start as documented so it can be fixed.

## Image structure

`Dockerfile` is a two-stage build:

1. **`builder`** — `python:3.11-slim-bookworm` + a venv with `pip install -r requirements.txt`.
   `onnxruntime`, `faiss-cpu`, and `opencv-python-headless` all ship prebuilt manylinux wheels for
   this platform/Python version, so no compiler toolchain is needed at all.
2. **`runtime`** — a fresh slim base, `libgl1` + `libglib2.0-0` installed (a long-documented
   requirement for `opencv-python-headless` to import cleanly even without any GUI use — its
   compiled extension still resolves those shared library symbols), the venv copied in from the
   builder stage, application code + `scripts/` copied in, and a non-root `appuser` (uid 1000) runs
   the process.

Nothing in either stage downloads anything at request time or at container startup beyond the
one-time `pip install` during the image build — **verified by grepping the codebase for any
download-style call (`requests.get`, `urlretrieve`, `wget`, `curl`, `huggingface_hub`, etc.); there
are none.** Models are loaded from local disk exactly once per process, at FastAPI startup
(`app/main.py`'s `lifespan`), and never re-read from disk on a per-request basis.

## Models: include or securely mount

Model binaries (`det_500m.onnx`, `w600k_mbf.onnx`) are **not** baked into the image by default —
they're large, environment/version-specific, and already excluded from version control
(`.gitignore`, `.dockerignore`). Two supported approaches:

**Recommended: mount read-only at runtime** (what `docker-compose.yml` does):

```bash
docker run -d \
  -v "$(pwd)/models:/app/models:ro" \
  -v face-verification-storage:/app/storage \
  --env-file .env \
  -p 8000:8000 \
  face-verification-api:local
```

`:ro` means the container can never modify the model files it was given — a real security
property, not just a convention (see `SECURITY.md` → "Model files are never exposed").

**Alternative: bake models into the image** (a self-contained image, at the cost of a larger image
and a rebuild required for every model update):

```dockerfile
# Add near the end of the runtime stage, after the existing `mkdir -p models ...` line:
COPY models/det_500m.onnx models/w600k_mbf.onnx ./models/
```

Only do this from a build context that actually has the real model files (they're
`.dockerignore`d by default specifically to prevent accidentally shipping stale/wrong binaries in
a generic build — remove the `models/*.onnx` line from `.dockerignore` locally if you take this
approach, and be deliberate about which model version ends up baked into which image tag).

## Persistence: FAISS index and metadata

`storage/faiss/` and `storage/metadata/` must be a **persistent volume**, not the container's
writable layer — otherwise every enrolled identity is lost when the container is recreated (a
redeploy, a crash restart, a host reboot). `docker-compose.yml` uses a named volume
(`face-verification-storage`); the equivalent with a bind mount:

```bash
docker run -d \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/data/storage:/app/storage" \
  --env-file .env \
  -p 8000:8000 \
  face-verification-api:local
```

`VectorStore.save()` writes atomically (temp file + rename) with restricted permissions, so a
container kill mid-write cannot corrupt the persisted files — see `SECURITY.md` → "FAISS index and
metadata storage" for the full detail.

## Health checks

The image's `HEALTHCHECK` calls `GET /health/ready` (not `/health/live`) — it only reports
`healthy` once both ONNX models have actually finished loading and the vector store is reachable,
which is what you want an orchestrator to gate real traffic on. `--start-period=60s` gives model
loading a grace window before failures count toward `unhealthy`/restart decisions.

If deploying to Kubernetes instead of plain Docker/Compose, map the two health endpoints to their
distinct k8s probe types rather than reusing one for both:

```yaml
livenessProbe:
  httpGet: { path: /health/live, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 15
readinessProbe:
  httpGet: { path: /health/ready, port: 8000 }
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 6   # tolerate model-loading time without flapping
```

Liveness should almost never fail once the process is up (it doesn't depend on models); readiness
is what should gate whether the pod receives traffic.

## Production ASGI startup configuration

The container's entrypoint (`docker/entrypoint.sh`) runs:

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --no-server-header \
  --timeout-keep-alive 30 --timeout-graceful-shutdown 30
```

Two important knobs are read directly by uvicorn from the environment rather than hardcoded:

- **`WEB_CONCURRENCY`** — number of worker processes (uvicorn's own built-in env var; defaults to
  `1` if unset). **Each worker process loads its own independent copy of both ONNX models** — this
  is correct/expected (it's what "models loaded once per process" means), but it also means total
  RAM usage scales roughly linearly with worker count. See "CPU and RAM requirements" below before
  choosing a value.
- **`FORWARDED_ALLOW_IPS`** — comma-separated IPs/CIDRs trusted to set `X-Forwarded-For` /
  `X-Forwarded-Proto` (defaults to `127.0.0.1`). Set this to your actual reverse proxy's address —
  needed for the rate limiter (`app/api/rate_limiting.py`) to see the real client IP instead of the
  proxy's, and for the app to correctly know it's being served over HTTPS behind TLS termination.
  Never set this to `*` unless every network hop in front of the container is fully trusted.

**Concurrency model — read before assuming this scales like a typical I/O-bound API**: route
handlers call into `app/services/*`, which perform CPU-bound ONNX inference *synchronously* inside
the request coroutine. This means a single worker process handles one recognition request's
inference at a time — other requests queue behind it (though FastAPI's own async plumbing, request
parsing, and lightweight endpoints like `/health/live` are unaffected). **The scaling lever here is
worker *processes*, not asyncio concurrency within one process**: set `WEB_CONCURRENCY` to roughly
the number of available vCPUs, and pair it with `ONNX_INTRA_OP_THREADS=1` /
`ONNX_INTER_OP_THREADS=1` so worker processes don't oversubscribe CPU cores against each other
(the default `0` lets ONNX Runtime pick its own thread count per session, which is *tuned for one
process owning the whole machine*, not for N processes sharing it). Load-test with your actual
hardware and traffic pattern before finalizing these numbers.

Other tunables worth knowing about (set via container env vars, all standard uvicorn flags/env
vars — see `uvicorn --help` for the full list): `REQUEST_TIMEOUT_SECONDS` (this app's own
timeout middleware, `app/api/timeout_middleware.py` — distinct from uvicorn's keep-alive timeout),
`--limit-concurrency` (caps in-flight requests, returning 503 beyond it — a blunter tool than the
app's own rate limiter, useful as a last-resort backstop).

## CPU and RAM requirements

These are **starting-point estimates**, not guarantees — actual figures depend on your CPU
architecture, ONNX Runtime version, and traffic pattern. Profile with your real hardware and
request volume before finalizing production sizing.

| | Per worker process |
|---|---|
| Minimum (functional, not performant) | 1 vCPU, 512MB–1GB RAM |
| Recommended baseline | 2 vCPU, 1–2GB RAM |

Why: `det_500m.onnx` (~2.5MB, SCRFD) and `w600k_mbf.onnx` (~13MB, MobileFaceNet) are both small,
CPU-efficient models individually, but each loaded ONNX Runtime `InferenceSession` carries its own
allocator/thread-pool overhead on top of the model weights, and the Python/FastAPI/OpenCV/NumPy
baseline itself typically sits in the 150–250MB range before any model is loaded. Budget total
container memory as roughly:

```
total_RAM ≈ (WEB_CONCURRENCY × per_worker_RAM) + a safety margin (~20%)
```

A container with `WEB_CONCURRENCY=4` and the recommended 1–2GB/worker baseline should be budgeted
4–8GB total, not 1–2GB — this is the single most common Docker sizing mistake for this kind of
service (assuming worker count is "free" the way it often is for lightweight I/O-bound workers).

## Index backup/recovery

**What to back up**: `storage/faiss/index.faiss` and `storage/metadata/metadata.json` together —
they're two halves of one consistent state (see `SECURITY.md` and `app/core/vector_store.py` for
why they're cross-validated against each other on load). Back them up as a pair, ideally from a
consistent point in time.

**Backup approach**: since writes are atomic (temp file + rename), a straightforward filesystem
snapshot or `cp`/`rsync` of both files while the service is running is safe — you will never catch
a torn/partial write mid-copy, though you could catch the two files a few milliseconds apart if an
enrollment happens mid-backup (acceptable for routine backups; wrap in an app-level "no writes
right now" window only if you need point-in-time consistency down to the millisecond, e.g. before a
major migration).

```bash
# From the host, against the named volume used by docker-compose.yml:
docker run --rm -v face-verification-storage:/data -v "$(pwd)/backup":/backup \
  alpine tar czf /backup/face-store-$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
```

**Recovery**: stop the container, restore both files (or the whole volume) to their expected
paths, then start the container back up. `VectorStore.load()` runs automatically at startup and
will use whatever's present — if the two files disagree with each other (e.g. you restored an old
metadata file against a newer index), it detects the inconsistency and resets to an empty store
rather than serving corrupted state, so restore both files from the *same* backup, never mixed.

```bash
docker compose down
docker run --rm -v face-verification-storage:/data -v "$(pwd)/backup":/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/face-store-YYYYMMDD-HHMMSS.tar.gz -C /data"
docker compose up -d
```

**Retention**: pair backups with `scripts/purge_stale_enrollments.py` (see `SECURITY.md` →
"Biometric data minimization") so backups themselves don't become an unbounded, unmanaged copy of
biometric data older than your retention policy allows.

## Model versioning

This service doesn't version models internally — it always loads whatever file is at
`DETECTOR_MODEL_PATH` / `RECOGNIZER_MODEL_PATH`. Recommended practice:

- **Name model files with a version in the filename** (e.g. `det_500m_v1.onnx`,
  `w600k_mbf_v2.onnx`) rather than always overwriting `det_500m.onnx` in place, and point the env
  vars at the specific versioned file. This makes rollback a one-line env var change, not a file
  hunt.
- **A changed embedding model invalidates the existing FAISS index.** Embeddings from different
  model versions are not comparable to each other — swapping `w600k_mbf.onnx` for a retrained or
  different-dimension version means every previously enrolled identity must be re-enrolled from
  scratch; there is no in-place migration path, because there's no way to recompute old embeddings
  without the original images. `app/main.py` already logs a warning
  (`embedding_dimension_mismatch`) if the loaded model's actual output dimension doesn't match
  `EMBEDDING_DIMENSION` — treat that warning as a hard signal to stop and re-enroll, not something
  to silence.
- **A changed detector model** (`det_500m.onnx`) is lower-risk — it doesn't affect stored
  embeddings, only future detection behavior (box/landmark quality), so a swap doesn't require
  re-enrollment. Still validate with `scripts/run_benchmark.py` against a labeled dataset before
  rolling out (see `README.md` → "Accuracy evaluation") — a detector change can shift accuracy even
  without touching the embedding model.
- **Tag container images with the model version(s) they were built/configured for** (e.g. an image
  tag or label recording which model filenames it expects) so a rollback of the container image and
  a rollback of the model files stay in sync, rather than becoming two separately-tracked things
  that can drift apart.

## Local Docker testing

```bash
# 1. Place real model files (not committed to the repo) here first:
#    models/det_500m.onnx
#    models/w600k_mbf.onnx

# 2. Set a real API key for this local run (docker-compose.yml requires it):
export FACE_API_KEY="local-testing-key-change-me"

# 3. Build and start
docker compose up --build

# 4. In another shell: confirm liveness, then readiness (should flip to 200
#    once models finish loading -- give it a few seconds)
curl -s http://localhost:8000/health/live
curl -s http://localhost:8000/health/ready

# 5. Exercise a real endpoint (requires the API key set above)
curl -s -X POST http://localhost:8000/api/v1/faces/verify \
  -H "X-API-Key: $FACE_API_KEY" \
  -F "file=@/path/to/a/test/photo.jpg"

# 6. Tear down (add -v to also remove the named storage volume)
docker compose down
```

Building and running without compose, directly with `docker`:

```bash
docker build -t face-verification-api:local .

docker run --rm -d --name face-verification-api \
  -p 8000:8000 \
  -e API_KEY_ENABLED=true -e API_KEY=local-testing-key-change-me \
  -e DETECTOR_MODEL_PATH=models/det_500m.onnx \
  -e RECOGNIZER_MODEL_PATH=models/w600k_mbf.onnx \
  -v "$(pwd)/models:/app/models:ro" \
  -v face-verification-storage:/app/storage \
  face-verification-api:local

docker logs -f face-verification-api   # watch startup / model loading
docker inspect --format='{{json .State.Health}}' face-verification-api  # check HEALTHCHECK status
docker stop face-verification-api
```

Running the existing automated test suite *inside* a container (useful for confirming the image's
Python/library versions behave the same as your dev environment — the suite itself needs no
model files, per `README.md` → "Testing"):

```bash
docker build --target builder -t face-verification-api:test-deps .
docker run --rm -v "$(pwd):/app" -w /app face-verification-api:test-deps \
  sh -c "pip install --quiet -r requirements-dev.txt && pytest -q"
```
