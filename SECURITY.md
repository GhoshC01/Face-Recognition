# Security & Privacy

This document describes how the Face Verification API is hardened for production use inside an
HRMS environment, and states explicitly what's implemented, what's a deployment responsibility,
and what trade-offs were made. It complements `README.md` (architecture/behavior) and
`docs/hrms-integration-plan.md` (the future HRMS-side integration contract).

## Authentication — separate from face recognition logic

`X-API-Key` header authentication is enforced by `require_api_key` (`app/api/deps.py`), attached
as a router-level FastAPI dependency on every `/api/v1/*` route. It is controlled by
`API_KEY_ENABLED` / `API_KEY` and knows nothing about faces, embeddings, or FAISS — `app/core/*`
and `app/services/*` have zero awareness that authentication exists at all. This separation is
deliberate: recognition logic stays testable and reusable independent of how (or whether) a given
deployment chooses to authenticate callers, and auth can be swapped (e.g. for mTLS or a gateway-
issued JWT) without touching a single line of recognition code.

**Startup guard**: if `API_KEY_ENABLED=true` and `API_KEY` is still the default placeholder
`"changeme"`, the service refuses to start when `ENVIRONMENT=production` (raises at startup), and
logs a warning in other environments. This exists specifically to prevent a real deployment from
silently running with no effective authentication.

Health/readiness endpoints (`/health/live`, `/health/ready`) are intentionally **not**
authenticated — this is the standard pattern for infra liveness/readiness probes (e.g. Kubernetes),
which typically cannot attach custom headers. `/health/ready` exposes an `enrolled_identities`
count; if this service is ever reachable from outside a trusted network, treat that as a minor
information disclosure and restrict network access accordingly rather than relying on route-level
auth for these two paths.

**Assumption**: authentication here is a shared-secret model appropriate for service-to-service
traffic (HRMS backend → this API), not end-user-facing auth. See `docs/hrms-integration-plan.md`
§4 for the full recommendation (secrets-manager-sourced key, network isolation as the primary
control, API key as defense-in-depth).

## Transport security (HTTPS/TLS)

This application does not terminate TLS itself in the primary recommended deployment — **TLS
termination at a reverse proxy / ingress / load balancer in front of the service is the expected
setup**, consistent with how most containerized Python services are deployed. Set
`ENFORCE_HTTPS=true` only if the app itself must redirect/enforce HTTPS (e.g. a standalone
deployment without a proxy); this adds Starlette's `HTTPSRedirectMiddleware`. Do not enable it
behind a proxy that only forwards plain HTTP internally, or all internal traffic will be redirected
incorrectly.

## Upload validation

Every file upload across all routes (`/enrollment`, `/faces/enroll`, `/faces/verify`,
`/faces/verify-multi`, `/verification/*`) is validated by `read_validated_upload`
(`app/api/upload_validation.py`) **before** any image decoding is attempted:

- **Content type** is checked against `ALLOWED_CONTENT_TYPES` (default: `image/jpeg`, `image/png`,
  `image/webp`) — rejected with `415 unsupported_media_type`.
- **Size** is checked against `MAX_UPLOAD_SIZE_MB` (default 8MB) — rejected with
  `413 payload_too_large`. Both the client-declared size and the actual bytes read are checked,
  since a client can misreport the former.
- **Malformed content** that passes both checks but isn't decodable as an image is rejected by
  `decode_image_bytes` with `400 invalid_image`.

In `/faces/verify-multi`, an individual bad frame (wrong type, oversized, or corrupt) is treated
the same as any other invalid frame — it's excluded, not fatal to the whole request.

**Deployment note**: application-level size checking happens after the body has already been
received and buffered by the ASGI server's multipart parser — it prevents *processing* an
oversized file but not the network/memory cost of *receiving* one. Set a body size cap at the
reverse proxy too (e.g. nginx `client_max_body_size`) for defense at the network layer.

## CORS

CORS is **off by default** (`CORS_ALLOW_ORIGINS=[]`) — no `CORSMiddleware` is even added to the
app unless origins are explicitly configured, since this is a server-to-server API and CORS is a
browser-only enforcement mechanism that server-to-server callers never need. If a browser-based
client does need direct access, set `CORS_ALLOW_ORIGINS` to an explicit list of origins. A `"*"`
wildcard origin is never combined with `allow_credentials=True` regardless of configuration — that
combination is rejected by browsers anyway and is a misconfiguration smell.

## Request IDs / correlation

`RequestContextMiddleware` (`app/api/middleware.py`) assigns a request id to every request
(reusing an inbound `X-Request-ID` if the caller supplies one), exposes it on the response via the
same header, and includes it in every structured log line and every error response body
(`ErrorResponse.request_id`). Forward this into HRMS's own logs to correlate a support
investigation across both systems — see `docs/hrms-integration-plan.md` §5.

## Logging: no raw biometric data

**Verified by auditing every `logger.*()` call in the codebase**: none pass a raw image, an aligned
face crop, or an embedding vector. Log lines only ever carry scalar/structural metadata —
`external_id`, `error_code`, `faiss_id`, byte counts, file paths, durations, timestamps. This is a
design invariant, not a filter bolted on afterward: `core/*` functions that could log something
useful for debugging (e.g. `QualityChecker`, `FaceRecognizer`) simply never had a reason to log
pixel or vector data in the first place, since every diagnostic need is already met by the
structured `QualityResult`/`ThresholdMetrics`-style objects returned to callers.

If you add new logging, keep this invariant: log *about* an image/embedding (its shape, source, an
error code) — never the image or embedding itself.

## Model files are never exposed

There is no static file mount anywhere in this application — `models/*.onnx` is read directly from
disk by `FaceDetector`/`FaceEmbedder` at startup and is never reachable via any HTTP route. Model
file paths never leak into API responses either: `FileNotFoundError` (raised when a model is
missing) is always caught and re-raised as `ModelNotReadyError`, whose message never includes the
path — only server-side logs do. A regression test (`tests/test_security_hardening.py::test_model_path_never_appears_in_error_response`)
asserts the configured model path substring never appears in an HTTP error response body.

**Deployment note**: keep `models/` outside any directory served by a reverse proxy or CDN, and
apply standard filesystem permissions (readable only by the service's runtime user).

## FAISS index and metadata storage

- The FAISS index (raw embedding vectors) and the JSON metadata sidecar (`faiss_id ↔ external_id`
  mapping, enrollment timestamps) are not exposed via any HTTP route — same reasoning as models.
- **Atomic, permission-restricted writes**: `VectorStore.save()` writes to a uniquely-named
  temporary file (`tempfile.mkstemp`) with `0o600` permissions before atomically renaming it into
  place — so a partially-written file is never left in the real path, and the file is never briefly
  world-readable while being written (best-effort; some platforms/filesystems don't honor POSIX
  permission bits, e.g. certain Windows configurations — the atomic rename still holds regardless).
- **Orphan cleanup**: any `.tmp_*` file left behind by a process that crashed mid-save in a
  previous run is removed the next time the store loads.
- **Corruption handling**: a missing or unreadable/unparseable index or metadata file is logged and
  treated as an empty store rather than crashing the process or operating on partial data; index
  and metadata vector counts are cross-checked on load and reset together if they disagree.
- **Deployment note**: `storage/faiss` and `storage/metadata` should sit on a volume with
  restricted OS-level permissions (readable/writable only by the service's runtime user) and,
  ideally, encryption at rest (e.g. an encrypted volume/disk) — FAISS's binary index format isn't
  application-level encrypted, so encryption-at-rest is a filesystem/infrastructure concern, not
  something this app implements itself.

## Secrets and configuration

All configuration — including `API_KEY` — is environment-variable-driven via `pydantic-settings`
(`app/config/settings.py`, `.env` / `.env.example`). Nothing is hardcoded. In production, source
`API_KEY` from your existing secrets manager rather than a plain `.env` file on disk.

## Rate limiting

`InMemoryRateLimiter` (`app/api/rate_limiting.py`), opt-in via `RATE_LIMIT_ENABLED` (default
`false`), enforces a sliding-window request budget (`RATE_LIMIT_REQUESTS` per
`RATE_LIMIT_WINDOW_SECONDS`) keyed by API key when present, else client IP — one registered caller
shares one budget regardless of which host it calls from.

**Explicit limitation**: this is in-memory and per-process. It does **not** correctly enforce a
global limit across multiple replicas or worker processes, since each keeps its own counters — a
deployment with N replicas effectively allows up to N× the configured limit. For a
horizontally-scaled deployment, prefer rate limiting at the API gateway/ingress (which sees all
traffic) or a shared store (e.g. Redis-backed limiter). This middleware is a reasonable default for
a single-instance deployment and defense-in-depth alongside gateway-level limiting — not a
replacement for it at scale.

## Temporary file handling

- The main request pipeline (detect → quality → align → embed) never writes anything to disk —
  uploaded images are decoded and processed entirely in memory as numpy arrays, so there is no
  temporary image file to clean up or leak in the first place.
- `VectorStore.save()`'s temp files are handled as described above (unique names, restricted
  permissions, atomic rename, cleaned up on write failure and on next load if orphaned).
- The offline benchmark tool (`app/evaluation/benchmark.py`) builds its isolated evaluation gallery
  under `tempfile.mkdtemp()` and removes it in a `finally` block once the run completes, regardless
  of success or failure.

## Request timeouts

`RequestTimeoutMiddleware` (`app/api/timeout_middleware.py`), always active, bounds every request
to `REQUEST_TIMEOUT_SECONDS` (default 30s) and returns `504 request_timeout` if exceeded — guarding
against a pathological image or a stalled ONNX call tying up a worker indefinitely. Tune this
higher if running on constrained CPU hardware where inference is slower, or lower for a
tighter SLA.

## Health / readiness checks

- `GET /health/live` — process-is-running check, always 200 if the app can respond at all.
- `GET /health/ready` — 200 only once both ONNX models have loaded successfully and the vector
  store is reachable; 503 otherwise. Use this (not liveness) to gate whether an orchestrator routes
  real traffic to an instance.

## Biometric data minimization

- **Embeddings are always retained**; raw enrollment photos are *also* retained by default as of
  `ENROLLMENT_IMAGES_DIR` (default `images/`) — a deliberate deviation from the original
  embeddings-only stance, added for operator/audit reference. Each accepted enrollment image is
  written to `<ENROLLMENT_IMAGES_DIR>/<external_id>/<embedding_id>.jpg`, best-effort (a disk failure
  here logs a warning but never rolls back an already-successful embedding enrollment, since the
  FAISS write remains the source of truth). Set `ENROLLMENT_IMAGES_DIR=""` to disable this and
  return to embeddings-only retention. The verification/identification request path is unaffected —
  those images are still processed entirely in memory and never written to disk (see "Temporary
  file handling" above).
- **Retention tooling**: `scripts/purge_stale_enrollments.py` lists (and, with `--delete`, removes)
  identities whose most recent enrollment predates a configurable retention window
  (`--older-than-days`), using `VectorStore.get_last_enrolled_at()`, and — when image persistence is
  enabled — also removes that identity's `<ENROLLMENT_IMAGES_DIR>/<external_id>/` folder so purged
  identities don't leave photos behind. This service has no automatic expiry on its own — HRMS
  decides who should remain enrolled (e.g. on offboarding, via
  `DELETE /api/v1/enrollment/{external_id}`, which also removes that identity's saved photos) — this
  script exists for periodic, policy-driven cleanup of anything that falls through that process.
- **What's genuinely retained**: FAISS vectors, an identity mapping, and (unless disabled) raw
  enrollment photos. Treat all of these as sensitive personal data under whatever regulatory
  framework applies (e.g. GDPR's special-category biometric data, or local equivalents) — this is a
  statement of what the system stores, not a substitute for a legal review of your specific
  deployment and jurisdiction. If raw-photo retention is not acceptable for your deployment, set
  `ENROLLMENT_IMAGES_DIR=""`.

## What's explicitly out of scope here

- End-user authentication/authorization (this API only authenticates the calling *system*, e.g.
  HRMS, not individual end users — that's HRMS's concern).
- Attendance business logic, employee business logic, and anything touching HRMS's own data —
  never implemented here, by design (see `README.md` → "What this service does NOT do").
- Legal/regulatory compliance review — the technical controls above support common privacy
  requirements (minimization, no raw-image retention, no biometric data in logs) but do not
  constitute legal advice or a compliance certification.
