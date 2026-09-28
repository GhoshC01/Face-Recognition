# Face Verification API

A reusable, ONNX-based face detection / recognition / verification service. It is deliberately
domain-agnostic: it stores only `external_id -> face embedding` mappings and returns a
similarity-based **PASS / FAIL** result. It has no knowledge of employees, attendance, or any
other caller-side business concept — that logic stays in the calling system (e.g. an HRMS API).

```
Client Application -> HRMS API -> Face Verification API -> Detection -> Alignment -> Embedding
                                                          -> FAISS -> Verification Result
                                -> HRMS API -> PASS/FAIL -> Attendance / Error (handled by HRMS)
```

## What this service does NOT do

- No attendance business logic, no attendance tables, no marking attendance.
- No HRMS employee business logic (leave, shifts, payroll, etc).
- `external_id` is an opaque string chosen entirely by the caller — this service assigns it no
  meaning beyond "a key that owns zero or more enrolled face embeddings."

HRMS integration itself is intentionally **not** part of this codebase yet — this service is being
completed and validated standalone first. See
[`docs/hrms-integration-plan.md`](docs/hrms-integration-plan.md) for the planned future contract,
auth/timeout recommendations, and a checklist for the eventual HRMS-side integration task.

## Architecture

```
app/
├── api/
│   ├── deps.py                 # FastAPI dependency providers (settings, services, auth)
│   ├── middleware.py           # request-id + structured access logging
│   ├── exception_handlers.py   # domain errors -> consistent JSON error responses
│   ├── upload_validation.py    # content-type/size checks on every upload (see SECURITY.md)
│   ├── rate_limiting.py        # opt-in in-memory sliding-window limiter
│   ├── timeout_middleware.py   # bounds request duration
│   └── routes/
│       ├── health.py           # unversioned /health/live, /health/ready
│       └── v1/
│           └── faces.py        # /api/v1/faces/compare-embedding
├── core/
│   ├── detector.py              # SCRFD ONNX face detector -> boxes + 5-point landmarks
│   ├── alignment.py              # 5-point similarity-transform warp to a configurable, model-sized crop
│   ├── quality.py                # structured, multi-reason quality gate (see below)
│   ├── embedding.py               # MobileFaceNet/ArcFace ONNX model -> raw embedding (dimension read from the model, never assumed)
│   ├── normalization.py            # L2-normalize a raw embedding (see below)
│   ├── recognizer.py               # detector -> quality -> alignment -> embedding -> normalization pipeline
│   ├── vector_store.py              # FAISS IndexFlatIP + JSON metadata sidecar (see below)
│   └── exceptions.py                 # domain error types
├── services/
│   ├── enrollment_service.py    # register a face against external_id
│   ├── verification_service.py  # 1:1 verify, 1:N identify, multi-frame consensus verification
│   └── evaluation_service.py    # stateless compare of two images, no storage involved
├── evaluation/                    # offline accuracy benchmarking -- see "Accuracy evaluation" below
│   ├── dataset.py                  # labeled gallery/probe dataset manifest loading
│   ├── metrics.py                   # accuracy/precision/recall/FAR/FRR/confusion matrix, threshold sweep
│   └── benchmark.py                  # runs the real pipeline over a dataset; isolated from production FAISS
├── schemas/                      # pydantic request/response contracts
├── config/                        # environment-driven settings + logging setup
├── utils/                          # image decoding, request-id context
└── main.py                          # app factory, lifespan (model loading), router wiring
```

See the top-level explanation in the project's design notes for the responsibility of each layer.

## Models

Place these in `models/` (see `models/README.md` for details):

- `det_500m.onnx` — SCRFD-500MF face detector.
- `w600k_mbf.onnx` — MobileFaceNet ArcFace embedding model (512-d).

These are **not** bundled with the repo (binary, environment-specific) — download them from your
model source of record and point `DETECTOR_MODEL_PATH` / `RECOGNIZER_MODEL_PATH` at them.

## Running locally

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements-dev.txt
cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

Without the ONNX model files present, the service still starts: `/health/live` returns 200, but
`/health/ready` returns 503 until both models load successfully.

## API overview (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/faces/compare-embedding` (form: `embedding`, `image`) | Compare one image against a supplied embedding. Returns `matched`, `status` (`Match` / `Not matching`), `confidence`, `threshold`, and image quality. Nothing is stored. |

Unversioned, for infra probes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Liveness |
| GET | `/health/ready` | Readiness (models loaded + vector store reachable) |

Interactive docs: `http://localhost:8000/docs`.

### Example: compare an embedding (`/faces/compare-embedding`)

```bash
curl -X POST http://localhost:8000/api/v1/faces/compare-embedding \
  -F "embedding=$(cat embedding.json)" \
  -F "image=@capture.jpg"
```

`embedding` is a JSON array of floats (or the same JSON as `0x`-prefixed hex). `image` must contain exactly one face. The response shape is unchanged: `matched`, `status` (`Match` or `Not matching`), `confidence`, `threshold`, `image` (detection score and quality), and `processed_at`.

## Quality module

`app/core/quality.py` runs after SCRFD detection and before MobileFaceNet embedding. It has no
knowledge of HRMS or attendance — it only judges whether a detected face crop is fit to embed, and
every check runs to completion so multiple simultaneous problems are reported together instead of
stopping at the first one:

```json
{"accepted": false, "quality_score": 0.42, "reasons": ["face_too_small", "image_too_blurry"]}
```

Checks performed, each independently configurable via `QualityThresholds` / env vars:

| Check | Reason code | Setting |
|---|---|---|
| Detection confidence | `low_detection_confidence` | `QUALITY_MIN_DETECTION_CONFIDENCE` |
| Crop validity (degenerate/out-of-bounds box) | `invalid_face_crop` | n/a (structural) |
| Minimum face size (pixels) | `face_too_small` | `QUALITY_MIN_FACE_WIDTH_PX` / `QUALITY_MIN_FACE_HEIGHT_PX` |
| Face size relative to frame | `face_area_ratio_too_low` | `QUALITY_MIN_FACE_AREA_RATIO` |
| Very dark image | `image_too_dark` | `QUALITY_MIN_BRIGHTNESS` |
| Overexposed image | `image_overexposed` | `QUALITY_MAX_BRIGHTNESS` |
| Blur / sharpness | `image_too_blurry` | `QUALITY_MIN_SHARPNESS` |

`quality_score` is a composite **image-quality** signal (brightness, sharpness, face-size ratio,
and detection confidence, equally weighted) — it is not a measure of recognition accuracy and
should not be confused with `similarity_score`. `accepted` is driven purely by the threshold
checks above, not by the score.

## Normalization module

`app/core/normalization.py` exposes one function, `l2_normalize(embedding)`, used by
`FaceEmbedder.embed()` right after MobileFaceNet produces a raw vector. It validates the vector
(rejects `None`/wrong-type/wrong-shape input, NaN/Inf values, and near-zero magnitude via
`InvalidEmbeddingError`), rescales it to unit length, and returns a vector of the same dimension.
It has no knowledge of FAISS, detection, or HRMS — it is a pure numeric transform reusable
anywhere a raw model output needs to become a comparison-ready embedding.

**Why normalize before storing/searching:** cosine similarity between vectors `a` and `b` is
`(a · b) / (|a| |b|)`. Once every vector is pre-normalized to `|v| = 1`, that formula collapses to
a plain dot product — exactly what an inner-product index (FAISS `IndexFlatIP`, used by
`vector_store.py`) computes natively, with no per-comparison division. This matters for three
reasons:

- **Correctness of the index type.** `IndexFlatIP` performs raw inner products. Without
  normalization, its "similarity" would be dominated by each vector's magnitude rather than its
  direction, silently breaking the cosine-similarity semantics the whole threshold system assumes.
- **Magnitude carries no identity signal.** An embedding's direction encodes the face; its raw
  magnitude can vary with lighting, pose, or model internals. Normalizing removes that
  irrelevant variance so comparisons reflect only facial similarity.
- **Stable, comparable thresholds.** With every vector on the same unit hypersphere, similarity
  scores always land in a fixed `[-1, 1]` range, so one configured `VERIFICATION_SIMILARITY_THRESHOLD`
  behaves consistently across every enrolled identity instead of drifting with unnormalized scale.

## Vector store

`app/core/vector_store.py` wraps FAISS `IndexIDMap2(IndexFlatIP(dimension))` behind a small,
reusable abstraction with no HRMS or attendance logic of any kind — it only maps opaque
`external_id`s to vectors:

| Method | Purpose |
|---|---|
| `add_embedding(external_id, embedding)` | Enroll one embedding under an identity; returns its FAISS id |
| `search(embedding, top_k)` | Rank enrolled identities by cosine similarity (raw scores; thresholding is the caller's job) |
| `remove_embedding(external_id)` | Remove every embedding enrolled under an identity |
| `get_embeddings(external_id)` / `has_identity(external_id)` | Read access for 1:1 verification |
| `save()` / `load()` | Persist to / restore from disk |
| `count()` | Total number of embedding vectors currently stored |

**Why `IndexFlatIP`:** with every vector L2-normalized (see Normalization above), inner product
*is* cosine similarity, so `IndexFlatIP` is the correct and efficient index type — no separate
distance conversion needed. `IndexIDMap2` (rather than plain `IndexFlatIP`) is used specifically
because it adds stable, externally-assigned integer ids and supports `remove_ids`, neither of
which the flat index provides on its own.

**Two files, not one:** the FAISS index (raw vectors) and a JSON metadata sidecar (the
`faiss_id <-> external_id` mapping, enrollment timestamps) are persisted separately. FAISS indexes
have no concept of arbitrary string keys or metadata, so the id-to-identity mapping has to live
outside the index regardless — keeping it as plain JSON also makes it trivially inspectable and
recoverable without FAISS tooling.

**Multiple embeddings per identity:** `external_to_ids` maps one `external_id` to a *list* of
FAISS ids, so an identity can be enrolled from several captures (e.g. different angles).
`get_embeddings()` returns all of them; `search()` collapses an identity's multiple ids down to
its single best-scoring match before ranking, so one identity never appears twice in a result set.

**Missing vs. corrupted files:** a missing index or metadata file (first run, or a wiped volume)
is normal and simply starts an empty store. A file that *exists* but fails to parse (truncated
FAISS index, invalid JSON, wrong vector dimension) is logged as a warning and treated the same as
missing — the store resets to empty rather than operating on partial data or crashing the process
that's loading it. The two files are also cross-checked against each other on load (vector count in
the index vs. entries in the metadata); a mismatch between them is treated as corruption too, since
serving stale or half-written state would be worse than starting clean. Saves are atomic
(write-to-temp-then-rename) to make that kind of corruption less likely to occur in the first
place.

## Configuration

All configuration is environment-based (`app/config/settings.py`, `pydantic-settings`). See
`.env.example` for the full list: model paths, detection/recognition thresholds, quality gate
bounds, storage locations, optional `X-API-Key` auth (`API_KEY_ENABLED`), CORS, rate limiting, and
request timeouts.

## Security & privacy

See [`SECURITY.md`](SECURITY.md) for the full production-hardening picture: authentication (kept
architecturally separate from recognition logic), upload validation, CORS defaults, rate limiting
and its scaling caveat, request timeouts, why model files and FAISS storage are never exposed over
HTTP, the "no raw images/embeddings in logs" invariant, temp-file safety, and biometric data
retention/minimization (including `scripts/purge_stale_enrollments.py`).

## Error handling

Domain errors (`NoFaceDetectedError`, `MultipleFacesDetectedError`, `LowImageQualityError`,
`IdentityNotFoundError`, ...) are raised from `core`/`services` and translated centrally in
`app/api/exception_handlers.py` into a consistent JSON shape:

```json
{
  "error_code": "no_face_detected",
  "message": "No face detected in the supplied image",
  "request_id": "5f3c2f8f9c1c4c6c9d9a5a2b6e6d6f61",
  "details": {},
  "timestamp": "2026-08-17T10:00:00Z"
}
```

Unhandled exceptions are caught, logged with a full traceback, and returned as a generic 500 so
internals are never leaked to callers.

## Accuracy evaluation (`app/evaluation/`, `scripts/run_benchmark.py`)

**Similarity score is not accuracy.** `similarity` is a raw cosine-similarity number between two
embeddings — a property of the model's output. *Accuracy* is a measured outcome: the percentage of
a labeled test set the whole pipeline got right, verified against ground truth. The two are easy to
conflate but answer different questions, so they're measured by entirely separate code paths in
this repo: `core/*` produces similarity scores at request time; `app/evaluation/` measures accuracy
offline, after the fact, against known-correct answers.

This is deliberately **not** a FastAPI endpoint. It's an offline tool for whoever operates the
service (not HRMS, not a live caller) to answer "how accurate is this system actually, right now,
at this threshold" — which requires a labeled dataset HRMS traffic doesn't provide.

```bash
python scripts/run_benchmark.py --dataset path/to/manifest.json
```

**Dataset**: a JSON manifest pointing at a *gallery* (reference images, one per identity, used to
build a lookup gallery — never mixed with production enrollment) and *probes* (labeled test/query
images with a known ground-truth `external_id`, or `null` for an impostor who isn't in the gallery
at all — required to measure False Accept Rate):

```json
{
  "gallery": [{"external_id": "EMP001", "image_path": "gallery/emp001.jpg"}],
  "probes": [
    {"external_id": "EMP001", "image_path": "probes/emp001_test1.jpg"},
    {"external_id": null, "image_path": "probes/stranger1.jpg"}
  ]
}
```

**Pipeline reuse**: `scripts/run_benchmark.py` constructs the exact same `FaceDetector` /
`FaceEmbedder` / `QualityChecker` / `FaceRecognizer` classes `app/main.py` wires up for the live
API, from the same settings — a benchmark run reflects what the deployed service actually does,
not a reimplementation of it.

**Production isolation**: by default, gallery images are enrolled into a brand-new, temporary
`VectorStore` that's discarded when the run finishes — production FAISS is never touched, read or
written. Passing `--use-production` switches to reading the already-enrolled production index as
the gallery instead (for evaluating the system as currently deployed) — even then, only
`search()` is ever called on it; nothing is enrolled or removed.

**What's recorded per probe** (`ProbeRecord`): `ground_truth_id`, `predicted_id` (top-1 gallery
match), `similarity`, `valid` (whether the pipeline could process the image at all), and
`rejection_reason` if not. PASS/FAIL and correctness are derived from these per threshold, without
re-running any recognition — which is what makes sweeping many thresholds cheap.

**Metrics** (`evaluate_at_threshold`, open-set identification with a reject option):

| Metric | Definition |
|---|---|
| Accuracy | `(correct) / total`, where correct = genuine probes matched to themselves + impostors correctly rejected |
| Precision | `TP / (TP + FP)` — of every accept the system made, how often it was right |
| Recall | `TP / (TP + FN)` — of every genuine person, how often they were correctly recognized |
| FAR (False Accept Rate) | impostor probes wrongly accepted ÷ total impostor probes — the strict biometric definition |
| FRR (False Reject Rate) | genuine probes not accepted (below threshold *or* pipeline failure) ÷ total genuine probes |
| Substitution error rate | genuine probes accepted as the *wrong* identity ÷ total genuine probes — a distinct failure mode FAR doesn't capture |
| Confusion matrix | `{ground_truth: {predicted_or_"<reject>": count}}`, with a synthetic `"<impostor>"` row for probes with no true identity — appropriate here specifically because this is an open-set, multi-identity, reject-capable system, not a binary classifier |

**Threshold selection**: `select_best_threshold` evaluates every candidate threshold you supply
and picks the one that maximizes a chosen objective (`accuracy` by default; `f1` or
`min_far_frr_gap` — an EER-like crossing point — also available) — **selected from measured
results on your labeled data, never a guessed constant.** The full sweep is always returned
alongside the winner so the accuracy/FAR/FRR tradeoff curve is inspectable, not just the final pick.

Example output, matching the shape of a typical benchmark summary:

```
At threshold=0.42 (selected by 'accuracy'):
  Total test images: 100
  Correct: 94
  Incorrect: 3
  Rejected/Unknown: 3
  Accuracy: 94.00%
  Precision: 96.91%  Recall: 96.91%
  FAR: 2.50%  FRR: 3.09%  Substitution errors: 1.03%
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests do not require real ONNX model files: core algorithms (alignment, quality gates, vector
store) are tested directly, and service-layer tests use a fake recognizer so enrollment/
verification/evaluation logic is exercised without ONNX Runtime.

## Reusability notes

Any application can call this service the same way HRMS does — it has no HRMS-specific
assumptions. `external_id` can be an employee code, a visitor badge id, a customer id, or anything
else the caller defines. Multiple embeddings can be enrolled per `external_id` (e.g. multiple
angles); verification takes the best match across all of them.

## Final architecture & production-readiness review

See [`docs/final-architecture-review.md`](docs/final-architecture-review.md) for a full,
evidence-based review against 11 categories (service boundary, detection, recognition, vector
search, enrollment, verification, HRMS integration status, accuracy, security, performance,
reusability) — including problems found (with severity and fixes, two of which were applied
directly), the final API contract, an architecture diagram, and a production-readiness checklist.
