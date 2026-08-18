# Final Architecture & Production-Readiness Review

Scope: the FaceVerification API as it exists in this repository today, plus its *planned* (not yet
implemented) HRMS integration. Every claim below was checked against the actual current code —
file/line references are given so each finding can be independently re-verified — not answered from
memory of having built it. Two small, low-risk issues found during this review were fixed directly
(noted inline); everything else is reported without code changes, per instruction.

---

## A. Service Boundary

| Question | Answer | Evidence |
|---|---|---|
| Is FaceVerification independent from HRMS? | **Yes** | No HRMS/employee/attendance code exists anywhere in `app/core`, `app/services`, or `app/evaluation`. Every mention of those words in that tree is a *docstring stating their absence* (verified by grep — see below), not actual logic. |
| Is attendance logic outside FaceVerification? | **Yes** | No attendance concept exists at all — not a field, not a table, not a status. `external_id` is the only identity concept, and it's opaque (the service assigns it no meaning). |
| Is employee business logic outside FaceVerification? | **Yes** | No employee model, no HR concepts (leave/shift/payroll), no database coupling of any kind — grepped for SQL/ORM drivers (`sqlalchemy`, `psycopg`, etc.); none found. The only persistence is FAISS + a JSON sidecar keyed by opaque `external_id`. |

Verified independence of the auth layer too: `grep -rn "from app.api" app/core app/services app/evaluation` returns nothing — `core`/`services`/`evaluation` never import from the API layer, so authentication, rate limiting, and upload validation cannot leak into recognition logic even by accident (see section I).

**Problems found: none.**

---

## B. Detection

| Question | Answer | Evidence |
|---|---|---|
| Is SCRFD loaded once? | **Yes** | `FaceDetector.load()` (`app/core/detector.py:107`) is a no-op if `self._session` is already set; `app/main.py`'s `lifespan` calls it exactly once at process startup. One ONNX session per worker **process** (not per-request, not globally across processes — see section J for what that distinction means for scaling). |
| Is detection configurable? | **Yes** | `input_size`, `confidence_threshold`, `nms_threshold`, `intra_op_threads`, `inter_op_threads` are all constructor parameters sourced from `Settings` (`DETECTOR_INPUT_SIZE`, `DETECTOR_CONFIDENCE_THRESHOLD`, `DETECTOR_NMS_THRESHOLD`, `ONNX_INTRA_OP_THREADS`, `ONNX_INTER_OP_THREADS`), not hardcoded. |
| Are landmarks used correctly? | **Yes** | 5-point SCRFD landmarks are decoded in the same coordinate space as the detected box, rescaled back to original-image coordinates via `/det_scale` (`detector.py:224`, applied identically to boxes and keypoints), and consumed by `align_face` in the exact order (`left-eye, right-eye, nose, left-mouth, right-mouth`) the ArcFace template (`alignment.py:14-23`) expects. `align_face` additionally validates landmark shape, finiteness, and non-degeneracy before use (`alignment.py:45-68`) rather than trusting the detector blindly. |

**Residual verification gap (Medium, informational, not a defect):** the SCRFD output-decoding logic (anchor generation, stride handling, output tensor ordering) was implemented against the *documented* InsightFace/SCRFD export convention, not verified numerically against the actual `det_500m.onnx` binary — this environment has never had the real model file available to run inference against. Recommend a one-time sanity check (a known test image with a known face) the first time real model weights are available, ideally wired into `scripts/run_benchmark.py`'s workflow.

**Problems found: none (residual verification gap noted above, not a code defect).**

---

## C. Recognition

| Question | Answer | Evidence |
|---|---|---|
| Is MobileFaceNet loaded once? | **Yes** | Same pattern as the detector: `FaceEmbedder.load()` (`embedding.py:71`) is idempotent, called once at startup. |
| Is model preprocessing correct? | **Yes, per documented convention** | `scalefactor=1/127.5, mean=(127.5,127.5,127.5), swapRB=True` (`embedding.py:122-128`) matches the standard ArcFace/MobileFaceNet preprocessing convention. Same residual caveat as detection: not numerically verified against the real binary. |
| Is actual embedding dimension verified? | **Yes** | `_infer_embedding_dimension()` (`embedding.py:16-31`) reads the dimension from the *loaded model's own output shape* at `load()` time — never a hardcoded assumption — and rejects a model with a non-fixed/dynamic output dimension. Every `embed()` call additionally re-checks the actual output length against that discovered dimension (`embedding.py:132-136`), raising `ModelNotReadyError` on drift. `app/main.py:62-69` also cross-checks the discovered dimension against the configured `EMBEDDING_DIMENSION` (used to size the FAISS index) and logs a warning on mismatch. |

**Problems found: none.**

---

## D. Vector Search

| Question | Answer | Evidence |
|---|---|---|
| Are embeddings normalized? | **Yes** | `FaceEmbedder.embed()` always returns `l2_normalize(embedding)` (`embedding.py:138`); `l2_normalize` (`app/core/normalization.py`) rejects NaN/Inf/near-zero vectors rather than silently producing a bad unit vector. |
| Is FAISS configured correctly for normalized vectors? | **Yes** | `IndexIDMap2(IndexFlatIP(dimension))` (`vector_store.py:65`) — inner product over unit vectors is exactly cosine similarity; `IndexIDMap2` additionally provides stable external ids and `remove_ids`, which plain `IndexFlatIP` lacks. |
| Is metadata mapping safe? | **Yes** | Two separate files (index + JSON sidecar), both written atomically (unique temp file → `chmod 0o600` → `os.replace`), cross-validated against each other on every `load()` (vector count must match; a mismatch resets both to empty rather than serving inconsistent state), corrupted/missing files handled without crashing, orphaned temp files from a crashed prior run swept on load. |
| Are multiple embeddings per identity supported? | **Yes** | `external_to_ids: dict[str, list[int]]` (`vector_store.py:29`) — an identity can own any number of embeddings; `search()` collapses them to that identity's single best-scoring match before ranking, so one identity never appears twice in a result set. |

**Problem found and fixed during this review (Medium severity):** `search()`, `get_embeddings()`, `has_identity()`, `list_identities()`, `get_last_enrolled_at()`, and `count()` read `self._index`/`self._state` **without** acquiring `self._lock`, while every mutating method (`add_embedding`, `remove_embedding`, `load`) does. This was not currently exploitable — the documented deployment model (section J) serializes all recognition work per worker process via a blocking call on the event loop, so no two threads ever touch the store concurrently *today*. But it was a latent race condition one refactor away from becoming real: `docs/deployment.md` itself floats offloading inference to a thread pool as a future throughput improvement, and doing that without this fix would have introduced a genuine read/write race on the FAISS index. **Fixed**: all six read methods now acquire the same lock as the write methods. Verified with the full test suite (179/179 still passing) — no behavior change, only added safety margin.

---

## E. Enrollment

| Question | Answer | Evidence |
|---|---|---|
| Are exactly two images supported? | **Yes** | `POST /api/v1/faces/enroll` declares `image1: UploadFile = File(...)` and `image2: UploadFile = File(...)` as required (`app/api/routes/v1/faces.py:19-22`) — omitting either yields FastAPI's own 422 validation error (tested: `test_faces_enroll_requires_both_images`). |
| Is one face required in each? | **Yes** | `enroll_pair()` calls `recognizer.process(image, strict_single_face=True)` for both images independently (`enrollment_service.py:69-70`) — zero faces raises `no_face_detected`, more than one raises `multiple_faces_detected`, for either image. |
| Is quality checked? | **Yes** | Same `recognizer.process()` call runs the full quality gate per image before any embedding is produced. |
| Are the two images checked for consistency? | **Yes** | Cosine similarity between the two resulting embeddings must clear `ENROLLMENT_MIN_IMAGE_SIMILARITY` (default 0.40) or the pair is rejected as `inconsistent_enrollment_images` (`enrollment_service.py:72-78`). |
| Is rollback safe? | **Yes** | Nothing is written to FAISS until *both* images pass detection, quality, and the consistency check (`enrollment_service.py:69-85` — a "validate then commit" ordering). The only remaining failure window — the second `add_embedding` FAISS write failing after the first succeeded — is caught explicitly and rolled back (`enrollment_service.py:87-96`). A previously-identified footgun in the "replace" duplicate policy (clearing the old enrollment *before* validating the new pair, which would have destroyed a valid enrollment on a failed retry) was already fixed in an earlier pass and has a dedicated regression test (`test_replace_policy_does_not_destroy_old_enrollment_on_validation_failure`). |

**Problem found, not fixed (Low severity, recommendation only):** nothing prevents `image1` and `image2` from being the exact same file. The consistency check trivially "passes" (identical embeddings → similarity 1.0), which silently defeats the intent of requiring two independent captures — someone could enroll from a single stolen photo submitted twice. This is a mild anti-fraud/liveness gap, not a correctness bug (the two-image requirement's stated purpose in the original spec was consistency-checking, which does technically hold for identical inputs). **Suggested fix** (not applied, per "don't rewrite unnecessarily"): compare a hash of the two raw uploads and reject an exact byte-for-byte duplicate; note this only catches the *identical file* case, not a re-photographed printout or a light re-crop — real liveness detection would need a different mechanism entirely and is out of scope for this service's current design.

---

## F. Verification

| Question | Answer | Evidence |
|---|---|---|
| Is PASS/FAIL threshold-based? | **Yes, throughout** | Every verification path (`verify_or_identify`, `verify_multi_frame`, the legacy `verify`/`identify`) computes `verified = similarity >= threshold` — no softmax, no top-1-always-wins logic anywhere. |
| Is Unknown supported? | **Yes** | Mode A (`/faces/verify` with no `external_id`) returns `external_id: null` + `status: FAIL` when nothing clears the identification threshold (`verification_service.py:130-143`) — an explicit "we don't know who this is" outcome, not a forced guess. Mode B against a claimed identity with zero enrolled embeddings returns `404 identity_not_found` — a distinct outcome from a similarity-based FAIL ("we have nothing to compare against" vs. "the face didn't match"). |
| Is the nearest result ever forced to PASS? | **No — verified by dedicated tests** | `test_verify_mode_a_fail_returns_null_external_id`, `test_verify_mode_a_with_nothing_enrolled_returns_fail`, and the multi-frame equivalents all assert that an unrelated/dissimilar probe FAILs even though it is technically the "closest available" candidate. |
| Is similarity returned? | **Yes, in every response schema** | `FaceVerificationResponse.similarity`, `VerificationResponse.similarity_score`, `MultiFrameVerificationResponse.similarity`, `CompareResponse.similarity_score` — always present, PASS or FAIL. |

**Observation (Low, not a defect):** the older `/api/v1/verification/verify` and `/verification/identify` endpoints use `strict_single_face=False` (pick the largest face when several are present), while the newer "main" `/api/v1/faces/verify` requires exactly one face. This is a deliberate, documented distinction (generic reusable primitives vs. the stricter HRMS-facing endpoint), but an integrator skimming the OpenAPI docs without reading the README could pick the wrong one for a security-sensitive flow. No code problem; worth keeping the docstrings/OpenAPI descriptions (already present) prominent.

**Problems found: none (one documented design distinction noted above).**

---

## G. HRMS Integration

**This section cannot be answered "yes" today, by design** — per explicit instruction in an earlier prompt, HRMS integration was deliberately deferred and never implemented in this codebase. Answering honestly:

| Question | Current state |
|---|---|
| Does HRMS call FaceVerification? | **Not yet — no HRMS code exists in this workspace.** The contract HRMS *would* call against is fully built and tested (`/api/v1/faces/verify`, `/api/v1/faces/verify-multi`, `/api/v1/faces/enroll`), but nothing currently calls it from an HRMS system. |
| Does HRMS receive PASS/FAIL? | **Contract exists, not wired.** Every verification response includes `status: "PASS"/"FAIL"` — ready to be consumed, not yet consumed. |
| Does HRMS decide whether to mark attendance? | **By design, this service has no opinion** — it has no attendance concept to decide about in the first place. Whether HRMS acts on PASS/FAIL is entirely future HRMS-side logic. |
| Is FaceVerification free from attendance logic? | **Yes** — see section A. |

The planned integration contract, auth recommendation, timeout/retry guidance, and an HRMS-side inspection checklist are documented in `docs/hrms-integration-plan.md`. Nothing in this review changes that plan — it's still accurate.

**Problems found: N/A — not yet implemented, on schedule with prior instruction.**

---

## H. Accuracy

| Question | Answer | Evidence |
|---|---|---|
| Is there an independent test dataset? | **Yes** | `app/evaluation/dataset.py`'s manifest format separates *gallery* (reference/enrollment-style images) from *probes* (labeled test images with ground truth, including impostor probes marked `external_id: null`), and `BenchmarkRunner` builds an isolated, temporary FAISS gallery by default — production enrollment data is never used as the accuracy dataset, and production FAISS is never written to (only optionally read, opt-in via `--use-production`, `benchmark.py:56-70`). |
| Are Accuracy/FAR/FRR measured? | **Yes, plus Precision/Recall/confusion matrix/substitution-error-rate** | `app/evaluation/metrics.py::evaluate_at_threshold` — an open-set-identification-with-reject-option model, with FAR strictly scoped to impostor probes (the correct biometric definition) and genuine-person-matched-to-wrong-identity tracked separately as a distinct "substitution error" metric. |
| Is threshold selected from test results? | **Yes** | `select_best_threshold()` sweeps every candidate threshold supplied and picks the one maximizing a measured objective (`accuracy`, `f1`, or `min_far_frr_gap`) — never a guessed constant — and returns the full sweep alongside the winner so the tradeoff curve is inspectable. |

**Recommendation (Info, not a defect):** this capability exists as an operator-triggered CLI tool (`scripts/run_benchmark.py`), not an automated/scheduled process. If accuracy regressions over time (e.g. after a model swap) are a concern, consider running it on a schedule or as a gate before promoting a new model version — see `docs/deployment.md` → "Model versioning."

**Problems found: none.**

---

## I. Security

| Question | Answer | Evidence |
|---|---|---|
| Is HRMS authenticated? | **Mechanism exists and is verified separate from recognition logic** | `X-API-Key` checked by `require_api_key` (`app/api/deps.py`), attached as a router-level dependency — confirmed via import-graph grep that `core`/`services` never import from `app/api`, so recognition logic has zero awareness auth exists. Startup refuses to boot in `ENVIRONMENT=production` if the API key is still the default placeholder (`main.py:36-42`). |
| Are images/embeddings protected? | **Yes** | No raw image or embedding is ever written to disk (in-memory processing only) or logged (every `logger.*()` call in the codebase was re-audited this session — grep output attached in `SECURITY.md`; only scalar metadata is ever logged). FAISS index writes use `0o600` permissions + atomic rename. |
| Are model/index files protected? | **Yes** | No static file mount exists anywhere in the app; model paths never leak into HTTP error responses (`FileNotFoundError` is always translated to a path-free `ModelNotReadyError`) — this is regression-tested (`test_model_path_never_appears_in_error_response`). |
| Is HTTPS expected? | **Yes, at the infrastructure layer** | The app documents TLS termination at a reverse proxy as the primary model (`SECURITY.md` → "Transport security"), with an opt-in `ENFORCE_HTTPS` flag for redirect enforcement if the app must handle it directly. |

**Problems found: none new this session** (the prior security-hardening pass already closed the real gaps that existed — unenforced upload validation, an insecure CORS default, missing rate limiting/timeouts — all verified still in place by this review's re-reading of `main.py`, `upload_validation.py`, `rate_limiting.py`, `timeout_middleware.py`).

**Restated limitation (already documented, not new):** the in-memory rate limiter (`app/api/rate_limiting.py`) is per-process — a multi-replica deployment effectively allows N× the configured limit. This is explicitly called out in `SECURITY.md` as a known trade-off, not something newly discovered here.

---

## J. Performance

| Question | Answer | Evidence |
|---|---|---|
| Are models loaded once? | **Yes, once per process** — see sections B/C. |
| Is FAISS reused? | **Yes** | One `VectorStore` instance is created at startup and held in `app.state`, shared by every request across the process's lifetime — never reconstructed per request. |
| Is CPU inference efficient? | **As efficient as the chosen models allow, with one real architectural caveat below.** SCRFD-500MF and MobileFaceNet are both small, CPU-oriented models by design; ONNX Runtime thread counts are configurable (`ONNX_INTRA_OP_THREADS`/`ONNX_INTER_OP_THREADS`) rather than left to fight over cores when scaling via multiple workers. |
| Are unnecessary image conversions avoided? | **One redundant conversion found and fixed this review.** |

**Problem found and fixed during this review (Low severity):** `QualityChecker._brightness()` and `_sharpness()` each independently called `cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)` on the *same* crop within a single `evaluate()` call — a wasted grayscale conversion on every single face processed. **Fixed**: `evaluate()` now converts to grayscale once and passes the result to both methods (`quality.py`). No behavior change (identical output), verified by the existing `test_quality.py` suite (all still passing, no test needed updating since all assertions go through the public `evaluate()` API).

**Architectural finding, documented not fixed (Medium severity, by design choice):** route handlers perform CPU-bound ONNX inference **synchronously inside `async def` request handlers**, without offloading to a thread pool (e.g. `starlette.concurrency.run_in_threadpool`). This means one worker process can only run one recognition request's inference at a time — concurrent recognition requests queue behind each other on that process's event loop (lightweight endpoints like `/health/live` are largely unaffected since they don't touch the blocking path). This is not a newly-discovered bug — it's an accepted, explicitly documented trade-off (`docs/deployment.md` → "Concurrency model"), with the prescribed mitigation being to scale via `WEB_CONCURRENCY` worker *processes* (one per vCPU) rather than in-process concurrency. This review's VectorStore locking fix (section D) is what makes the *alternative* mitigation (thread-pool offloading within one process) safe to adopt later, if desired — it wasn't safe to recommend before this fix.

---

## K. Reusability

| Question | Answer | Evidence |
|---|---|---|
| Can another application call FaceVerification without knowing its internals? | **Yes** | The public contract is HTTP + JSON (multipart uploads, JSON responses) — no client needs to know ONNX, FAISS, or Python exist behind it. `external_id` is caller-defined and opaque; nothing HRMS-specific leaks into the contract (see section F below for the full endpoint list). |
| Can the recognition model be replaced later? | **Yes, without touching calling code** | `FaceEmbedder` never assumes an embedding dimension — it reads it from whatever model is loaded (section C) — and `FaceDetector`/`FaceEmbedder` are constructed from file paths in `Settings`, not imported by name elsewhere. Swapping model files (with matching preprocessing conventions) requires no code change, only a settings change and, per `docs/deployment.md` → "Model versioning," re-enrollment if the embedding model/dimension changes. |
| Can FAISS be replaced later? | **Architecturally yes, with moderate, contained effort.** `VectorStore` is the only class that imports `faiss` directly (confirmed via grep — no other file imports the `faiss` package). Every service (`EnrollmentService`, `VerificationService`, `EvaluationService`) depends on `VectorStore`'s *interface* (`add_embedding`, `search`, `remove_embedding`, `get_embeddings`, `count`, ...), not on FAISS specifics. A different vector backend (e.g. a managed vector DB) would mean rewriting `VectorStore` internally while keeping its method signatures — no changes needed above that layer. |
| Can HRMS be removed without rewriting the core recognition engine? | **Trivially yes — there is currently nothing to remove.** HRMS integration doesn't exist in this codebase (section G); the recognition engine (`core/*`) was built and tested entirely standalone, with HRMS integration planned as a pure *addition* on top (a caller, not a dependency) per `docs/hrms-integration-plan.md`. |

**Problems found: none.**

---

## Consolidated Problems Found

| # | Problem | Section | Severity | Status |
|---|---|---|---|---|
| 1 | `QualityChecker` computed grayscale conversion twice per face (once each for brightness/sharpness) | J | Low | **Fixed this review** (`app/core/quality.py`) |
| 2 | `VectorStore` read methods (`search`, `get_embeddings`, `has_identity`, `list_identities`, `get_last_enrolled_at`, `count`) were not lock-protected, unlike writes — latent race condition if thread-based concurrency is introduced later | D / J | Medium | **Fixed this review** (`app/core/vector_store.py`) |
| 3 | No detection of byte-identical `image1`/`image2` in two-image enrollment — a single photo submitted twice trivially "passes" the consistency check | E | Low | Reported, not fixed (design recommendation only) |
| 4 | Route handlers block the event loop with synchronous CPU-bound inference; throughput scales only via worker processes, not in-process concurrency | J | Medium | Documented trade-off (`docs/deployment.md`), not a defect — mitigation available now that #2 is fixed |
| 5 | SCRFD/MobileFaceNet preprocessing and output-decoding were implemented against documented conventions, never numerically verified against the real `.onnx` binaries (unavailable in this environment) | B / C | Medium | Residual verification gap — recommend a sanity check on first real deployment |
| 6 | In-memory rate limiter is per-process; under-enforces across multiple replicas | I | Medium | Documented trade-off (`SECURITY.md`), not new |
| 7 | Two verification endpoint families with different single/multi-face strictness policies could confuse an integrator | F | Low | Documented distinction, no code issue |

No High or Critical severity issues were found. Nothing in this list required a rewrite of working logic — items 1 and 2 were minimal, behavior-preserving patches (verified via the full existing test suite, no tests needed modification); the rest are either accepted trade-offs already documented elsewhere, forward-looking recommendations, or informational.

---

## Final API Contract

Base path: `{API_V1_PREFIX}` (default `/api/v1`). All routes below (except health) require `X-API-Key` when `API_KEY_ENABLED=true`.

| Method | Path | Purpose | Key request fields | Key response fields |
|---|---|---|---|---|
| POST | `/faces/enroll` | **Primary enrollment**: two images, cross-validated | `external_id`, `image1`, `image2` (multipart) | `success`, `external_id`, `images_processed`, `enrollment_status`, `image_similarity`, `images[]` |
| POST | `/faces/verify` | **Primary verification** (HRMS-facing): 1:1 if `external_id` given, else 1:N | `file`, optional `external_id` | `verified`, `status` (`PASS`/`FAIL`), `external_id`, `similarity`, `threshold`, `mode`, `quality` |
| POST | `/faces/verify-multi` | Multi-frame consensus verification (webcam capture) | `files[]` (3–5), optional `external_id`, `debug` | `verified`, `status`, `external_id`, `similarity`, `frames_submitted/valid/agreeing`, `consensus_ratio`, optional `frames[]` |
| POST | `/enrollment` | Add a single embedding to an (optionally already-enrolled) identity | `external_id`, `file` | `enrolled`, `embedding_id`, `quality` |
| GET | `/enrollment/{external_id}` | Enrollment status | — | `enrolled`, `embedding_count` |
| DELETE | `/enrollment/{external_id}` | Remove all embeddings for an identity | — | `removed`, `embeddings_removed` |
| POST | `/verification/verify` | Legacy 1:1 verify (lenient face selection) | `external_id`, `file` | `verified`, `result`, `similarity_score`, `threshold` |
| POST | `/verification/identify` | Legacy 1:N identify, returns a ranked list | `file`, optional `top_k` | `matches[]` (`external_id`, `similarity_score`) |
| POST | `/verification/compare` | Stateless 1:1 compare, no enrollment involved | `file_a`, `file_b` | `match`, `result`, `similarity_score` |
| GET | `/health/live` | Liveness (unauthenticated) | — | `status` |
| GET | `/health/ready` | Readiness — 503 until models loaded (unauthenticated) | — | `status`, `detector_loaded`, `recognizer_loaded`, `vector_store_ready`, `enrolled_identities` |

**Universal error shape** (any non-2xx): `{error_code, message, request_id, details, timestamp}`. `error_code` is the stable, branchable field — see `SECURITY.md`/`README.md` for the full code list (`no_face_detected`, `multiple_faces_detected`, `low_image_quality`, `identity_not_found`, `identity_already_exists`, `inconsistent_enrollment_images`, `invalid_frame_count`, `unsupported_media_type`, `payload_too_large`, `model_not_ready`, `rate_limit_exceeded`, `request_timeout`, ...).

Offline, non-HTTP tooling (not part of the API contract, run via CLI): `scripts/run_benchmark.py` (accuracy evaluation), `scripts/purge_stale_enrollments.py` (retention).

---

## Final Architecture Diagram

```
                         Client Application
                                 |
                                 v
                     Existing HRMS API  (not yet built — see docs/hrms-integration-plan.md)
                                 |
                                 |  X-API-Key auth, HTTPS (terminated upstream)
                                 v
        +--------------------------------------------------------------+
        |                 FaceVerification API (this repo)              |
        |                                                                |
        |  api/          FastAPI routes, auth, upload validation,        |
        |                rate limiting, timeouts, request-id             |
        |                    |  (zero imports flow the other direction)  |
        |                    v                                           |
        |  services/     EnrollmentService, VerificationService,         |
        |                EvaluationService -- business rules for THIS    |
        |                service only (never attendance/employee rules)  |
        |                    |                                           |
        |                    v                                           |
        |  core/         FaceDetector (SCRFD) --> QualityChecker         |
        |                    --> align_face --> FaceEmbedder (Mobile-   |
        |                    FaceNet) --> l2_normalize --> VectorStore   |
        |                    (FAISS IndexIDMap2/IndexFlatIP + JSON        |
        |                    sidecar)                                    |
        |                                                                |
        |  evaluation/   Offline: dataset.py + metrics.py + benchmark.py |
        |                (isolated gallery, never touches prod FAISS      |
        |                by default)                                     |
        +--------------------------------------------------------------+
                                 |
                                 v
                     PASS / FAIL + similarity + threshold
                                 |
                                 v
                     Existing HRMS API  (future)
                                 |
                        +--------+---------+
                        v                  v
              Existing attendance    Error handling
              logic (HRMS-owned,      (HRMS-owned)
              never touched here)
```

Persistent state: `storage/faiss/index.faiss` + `storage/metadata/metadata.json` (one FAISS vector
store, reused across all requests in a process — never rebuilt per request). Models
(`models/det_500m.onnx`, `models/w600k_mbf.onnx`) are read once at process startup and never
re-read or re-downloaded.

---

## Production-Readiness Checklist

- [x] Service boundary: no attendance/employee logic anywhere in the codebase (verified by grep, not just docstring claims)
- [x] SCRFD + MobileFaceNet each loaded exactly once per worker process, never per-request
- [x] Detection and recognition thresholds fully configurable via environment variables
- [x] Embedding dimension read from the model at load time, never hardcoded/assumed
- [x] Embeddings L2-normalized; FAISS configured as `IndexIDMap2(IndexFlatIP)` to match
- [x] Metadata mapping safe: atomic writes, corruption-tolerant loads, cross-validated against the index
- [x] Multiple embeddings per identity supported end-to-end
- [x] Two-image enrollment: one face each, quality-gated, cross-checked, rollback-safe
- [x] PASS/FAIL always threshold-based; Unknown/reject always available; nearest result never forced to PASS
- [x] Similarity + threshold always returned, on both PASS and FAIL
- [x] Independent accuracy evaluation tooling: separate dataset, Accuracy/Precision/Recall/FAR/FRR/confusion matrix, threshold selected from measured results
- [x] Authentication implemented and architecturally isolated from recognition logic
- [x] Upload validation (type + size), CORS locked down by default, rate limiting and request timeouts in place
- [x] No raw images/embeddings ever logged or persisted; model/index files never exposed over HTTP
- [x] Containerized: non-root user, health-checked, models/storage externalized via mounts/volumes
- [x] Backup/recovery and model-versioning procedures documented
- [ ] **Not yet done**: HRMS integration itself (by design — deferred to a future task; plan is documented)
- [ ] **Not yet done**: numeric verification of detector/embedder preprocessing against the real `.onnx` binaries (blocked on this environment not having the model files)
- [ ] **Recommended before scaling out**: load-test to validate `WEB_CONCURRENCY` sizing; move rate limiting to gateway-level or a shared store if deploying more than one replica
- [ ] **Recommended before go-live**: run `scripts/run_benchmark.py` against a real labeled dataset to select production thresholds from measured data, not the shipped defaults
