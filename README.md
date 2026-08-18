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
│           ├── enrollment.py   # /api/v1/enrollment (single-image add/status/remove)
│           ├── faces.py        # /api/v1/faces/enroll, /faces/verify, /faces/verify-multi
│           └── verification.py # /api/v1/verification
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
`/health/ready` returns 503 until both models load successfully. This lets you deploy the
container and run infra checks before models are mounted, while still failing readiness probes
correctly.

## Running with Docker

```bash
export FACE_API_KEY="local-testing-key-change-me"   # docker-compose.yml requires this to be set
# place real det_500m.onnx / w600k_mbf.onnx under models/ first
docker compose up --build
```

See [`docs/deployment.md`](docs/deployment.md) for the full picture: image structure, mounting vs.
baking in models, persisting the FAISS index/metadata as a volume, health checks, production ASGI
startup (`WEB_CONCURRENCY`, `FORWARDED_ALLOW_IPS`, the CPU-bound concurrency model), CPU/RAM sizing
guidance, index backup/recovery, model versioning, and more detailed local testing instructions.

## API overview (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/enrollment` (form: `external_id`, `file`) | Enroll a single face image for an identity |
| GET | `/enrollment/{external_id}` | Enrollment status |
| DELETE | `/enrollment/{external_id}` | Remove all enrolled embeddings for an identity |
| POST | `/faces/enroll` (form: `external_id`, `image1`, `image2`) | Initial enrollment: two images, cross-validated as the same person |
| POST | `/faces/verify` (form: `file`, optional `external_id`) | **Main HRMS-facing endpoint** — Mode B (1:1) if `external_id` given, Mode A (1:N identify) if omitted |
| POST | `/faces/verify-multi` (form: `files[]`, optional `external_id`, `debug`) | Multi-frame verification for webcam capture — see below |
| POST | `/verification/verify` (form: `external_id`, `file`) | 1:1 verify — returns `PASS`/`FAIL` |
| POST | `/verification/identify` (form: `file`, optional `top_k`) | 1:N search across all enrolled identities |
| POST | `/verification/compare` (form: `file_a`, `file_b`) | Stateless 1:1 compare, no enrollment needed |

Unversioned, for infra probes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Liveness |
| GET | `/health/ready` | Readiness (models loaded + vector store reachable) |

Interactive docs: `http://localhost:8000/docs`.

### Example: initial enrollment (two images)

```bash
curl -X POST http://localhost:8000/api/v1/faces/enroll \
  -F "external_id=EMP001" \
  -F "image1=@front.jpg" \
  -F "image2=@side.jpg"
```

```json
{
  "success": true,
  "external_id": "EMP001",
  "images_processed": 2,
  "enrollment_status": "success",
  "image_similarity": 0.82,
  "images": [
    {"image": "image1", "embedding_id": 0, "detection_score": 0.99, "quality": {"accepted": true, "quality_score": 0.93, "reasons": [], "metrics": {"...": "..."}}},
    {"image": "image2", "embedding_id": 1, "detection_score": 0.98, "quality": {"accepted": true, "quality_score": 0.90, "reasons": [], "metrics": {"...": "..."}}}
  ],
  "message": "Face enrollment completed successfully"
}
```

`POST /api/v1/faces/enroll` is the two-image onboarding workflow — `POST /api/v1/enrollment`
(single image) still exists for adding further embeddings to an already-enrolled identity later.
The pipeline for each image is `SCRFD -> quality gate -> alignment -> MobileFaceNet -> L2 normalize`;
nothing is written to FAISS until *both* images have individually passed every check:

- **Exactly one face per image** — `strict_single_face=True` turns zero faces into
  `no_face_detected` (422) and more than one into `multiple_faces_detected` (422) for either image.
- **Quality gate** — each image runs through the same `QualityChecker` used everywhere else (see
  Quality module below); a failing image raises `low_image_quality` (422) with its `reasons`.
- **Cross-image consistency** — the two resulting embeddings must have cosine similarity ≥
  `ENROLLMENT_MIN_IMAGE_SIMILARITY` (default `0.40`), or the pair is rejected as
  `inconsistent_enrollment_images` (422) — this is the "are these actually the same person" check.
- **Duplicate policy** — `ENROLLMENT_DUPLICATE_POLICY` (default `reject`) rejects re-enrolling an
  `external_id` that already has embeddings with `identity_already_exists` (409). Setting it to
  `replace` clears the old enrollment instead — but only *after* the new pair has passed every
  check above, so a rejected re-enrollment attempt never costs the caller their previously valid
  enrollment.
- **No partial enrollment** — both embeddings are validated and normalized before either is
  written to FAISS. The only remaining failure window is the storage step itself (e.g. the second
  `add_embedding` call failing after the first succeeded); that case is caught explicitly and
  rolled back by removing whatever was just written, so a failed second write never leaves the
  identity half-enrolled.

### Example: main verification endpoint (`/faces/verify`)

This is the endpoint HRMS calls at attendance-capture time. It always runs the same pipeline —
`SCRFD (exactly one face required) -> quality gate -> alignment -> MobileFaceNet -> L2 normalize`
— then branches on whether `external_id` was supplied:

```bash
# Mode B: verify against a specific, claimed identity
curl -X POST http://localhost:8000/api/v1/faces/verify \
  -F "external_id=EMP001" \
  -F "file=@capture.jpg"
```

```json
{
  "verified": true,
  "status": "PASS",
  "external_id": "EMP001",
  "similarity": 0.91,
  "threshold": 0.85,
  "mode": "verification",
  "detection_score": 0.98,
  "quality": {"...": "..."},
  "processed_at": "2026-08-17T10:00:00Z"
}
```

```bash
# Mode A: no claimed identity -- identify the best match across everyone enrolled
curl -X POST http://localhost:8000/api/v1/faces/verify -F "file=@capture.jpg"
```

```json
{
  "verified": false,
  "status": "FAIL",
  "external_id": null,
  "similarity": 0.62,
  "threshold": 0.85,
  "mode": "identification",
  "detection_score": 0.97,
  "quality": {"...": "..."},
  "processed_at": "2026-08-17T10:00:00Z"
}
```

Notes on the contract:

- **Exactly one face is required** — unlike `/verification/verify`/`/verification/identify` (which
  pick the largest face when several are present), this endpoint raises `multiple_faces_detected`
  (422) for more than one face and `no_face_detected` (422) for zero, since it's the primary
  attendance-capture path where an ambiguous frame should never be silently resolved.
- **`external_id` is echoed back in Mode B regardless of PASS/FAIL** — the caller supplied it
  themselves, so returning it is just confirming what was checked, not asserting a match.
- **`external_id` is only populated in Mode A on PASS** — a low-confidence best-guess identity is
  never surfaced when the score doesn't clear the threshold; a FAIL always carries `external_id: null`.
- **The nearest candidate is never forced to PASS.** Both modes compare a real similarity score
  against a real configured threshold (`VERIFICATION_SIMILARITY_THRESHOLD` for Mode B,
  `IDENTIFICATION_SIMILARITY_THRESHOLD` for Mode A); an empty vector store or an unrelated face
  always resolves to FAIL, never to whatever happened to be closest.
- Mode B against an `external_id` with no enrolled embeddings returns `identity_not_found` (404),
  not FAIL — "we have nothing to compare against" is a different failure than "the face didn't match".
- This endpoint never marks attendance and never writes to HRMS data — it only returns a verdict;
  everything downstream of PASS/FAIL is HRMS's responsibility.

### Example: verify

```bash
curl -X POST http://localhost:8000/api/v1/verification/verify \
  -F "external_id=EMP-1024" \
  -F "file=@capture.jpg"
```

```json
{
  "external_id": "EMP-1024",
  "verified": true,
  "result": "PASS",
  "similarity_score": 0.71,
  "threshold": 0.36,
  "detection_score": 0.98,
  "quality": {
    "accepted": true,
    "quality_score": 0.91,
    "reasons": [],
    "metrics": {
      "detection_confidence": 0.98,
      "face_width": 210,
      "face_height": 224,
      "face_area_ratio": 0.18,
      "brightness": 132.4,
      "sharpness": 210.7
    }
  },
  "processed_at": "2026-08-17T10:00:00Z"
}
```

HRMS treats `result` as the sole verdict and owns everything downstream of it.

## Multi-frame verification (`/faces/verify-multi`)

Optional, for webcam-based attendance capture: instead of trusting one frame, HRMS submits several
(configurable, default 3–5) frames captured in quick succession, and this endpoint only PASSes
when enough of them independently agree — reducing false recognition caused by any single blurry,
partially occluded, or otherwise poor-quality frame.

```bash
curl -X POST http://localhost:8000/api/v1/faces/verify-multi \
  -F "external_id=EMP001" \
  -F "files=@frame1.jpg" -F "files=@frame2.jpg" -F "files=@frame3.jpg"
```

```json
{
  "verified": true,
  "status": "PASS",
  "external_id": "EMP001",
  "similarity": 0.91,
  "threshold": 0.85,
  "mode": "verification",
  "frames_submitted": 3,
  "frames_valid": 3,
  "frames_agreeing": 3,
  "consensus_ratio": 1.0,
  "required_consensus_ratio": 0.6,
  "reasons": [],
  "frames": null,
  "processed_at": "2026-08-17T10:00:00Z"
}
```

Add `-F "debug=true"` to get per-frame diagnostics instead of `"frames": null`:

```json
"frames": [
  {"frame_index": 0, "valid": true, "external_id": "EMP001", "similarity": 0.91, "passed_threshold": true, "detection_score": 0.98, "quality": {"...": "..."}, "rejection_reason": null},
  {"frame_index": 1, "valid": true, "external_id": "EMP001", "similarity": 0.89, "passed_threshold": true, "detection_score": 0.97, "quality": {"...": "..."}, "rejection_reason": null},
  {"frame_index": 2, "valid": true, "external_id": "EMP001", "similarity": 0.93, "passed_threshold": true, "detection_score": 0.99, "quality": {"...": "..."}, "rejection_reason": null}
]
```

How the verdict is decided (`app/services/verification_service.py::verify_multi_frame`):

1. **Each frame runs the same single-face pipeline independently** (`strict_single_face=True`).
   A frame that fails to decode, has no/multiple faces, or fails the quality gate is marked
   invalid with a `rejection_reason` (its own `error_code`) and simply excluded — it never fails
   the whole request.
2. **Not enough valid frames → FAIL, reason `insufficient_valid_frames`.** Configurable via
   `MULTI_FRAME_MIN_VALID_FRAMES` (default 2) — a request can't be salvaged from too few usable
   frames no matter how good they are.
3. **Identity consistency**: among the valid frames, whichever identity was the top match most
   often is the "leading candidate" (Mode B: this is always the claimed `external_id`; Mode A:
   different frames could in principle point at different people).
4. **Threshold agreement**: of the leading candidate's frames, count how many *also* individually
   cleared the similarity threshold (`frames_agreeing`), as a fraction of all valid frames
   (`consensus_ratio`).
5. **PASS requires both conditions plus an absolute floor**:
   `frames_agreeing >= MULTI_FRAME_MIN_AGREEING_FRAMES` (default 2) **and**
   `consensus_ratio >= MULTI_FRAME_CONSENSUS_RATIO` (default 0.6). The absolute floor exists
   specifically so **one strong (or lucky) frame can never carry a PASS by itself** — even a
   single frame with 100% "agreement" (1 out of 1) fails the floor check.
6. Like the single-frame endpoint: `external_id` is always echoed in Mode B (PASS or FAIL) and
   only populated in Mode A on PASS; the closest available candidate is never forced to PASS.

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
