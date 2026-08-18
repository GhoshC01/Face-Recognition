# Future HRMS Integration Plan

**Status: planning only.** No HRMS code has been inspected, modified, or created for this
document. Nothing in this file implements integration — it exists so that a later, explicit
integration task has a concrete starting point instead of reopening these decisions from scratch.
The FaceVerification API itself is already implemented and tested independently of HRMS; this
document describes how an *existing* HRMS application would call it once that integration work is
scheduled.

---

## 1. Future integration architecture

```
Client Application
      |
      v
Existing HRMS API  --------------------------------------------+
      |                                                         |
      | POST /api/v1/faces/verify (or /faces/enroll)            |
      | (image/frame [+ external_id])                           |
      v                                                         |
FaceVerification API                                            |
      |                                                         |
      v                                                         |
  SCRFD detection                                                |
      v                                                         |
  Quality validation                                             |
      v                                                         |
  Face alignment                                                 |
      v                                                         |
  MobileFaceNet embedding                                        |
      v                                                         |
  L2 normalization                                                |
      v                                                         |
  FAISS search / compare                                          |
      v                                                         |
  PASS / FAIL + similarity + threshold                            |
      |                                                         |
      +---------------------------------------------------------+
      |
      v
Existing HRMS API
      |
      +--> PASS --> Existing attendance/business logic (unchanged, HRMS-owned)
      |
      +--> FAIL / error --> Existing error-handling flow (unchanged, HRMS-owned)
```

Everything left of "Existing HRMS API" in the diagram already exists in this repository today —
`app/api/routes/v1/faces.py`, `app/core/*`, `app/services/*` — and is independently tested. The
only new component in a future integration task is the **HRMS-side caller**: an outbound HTTP
client inside HRMS that calls this API and interprets its response. Nothing changes on the
FaceVerification side unless section 8 below turns out to be necessary.

## 2. Future API contract: HRMS → FaceVerification

This is not hypothetical — it is the contract already implemented and tested in this repository.
HRMS integration work later means *consuming* these endpoints, not designing new ones from
scratch (barring anything identified in section 8).

| Use case | Method & path | Notes |
|---|---|---|
| Initial enrollment (onboarding) | `POST /api/v1/faces/enroll` | Two images, cross-validated as the same person. See `README.md` → "Initial enrollment". |
| Add a single additional embedding later | `POST /api/v1/enrollment` | e.g. re-enrolling one new angle without redoing the full two-image flow. |
| **Attendance-time verification (primary)** | `POST /api/v1/faces/verify` | The endpoint HRMS is expected to call at attendance-capture time. Supports both identity-known and identity-unknown flows in one call. |
| Check enrollment status | `GET /api/v1/enrollment/{external_id}` | Useful for HRMS to confirm an employee is enrolled before prompting for a capture. |
| Remove enrollment (offboarding) | `DELETE /api/v1/enrollment/{external_id}` | HRMS-triggered cleanup when an employee record is deactivated. |
| Liveness / readiness | `GET /health/live`, `GET /health/ready` | For HRMS's own health-check/monitoring wiring, and for confirming the service is ready (models loaded) before routing real traffic to it. |

`POST /api/v1/faces/verify` is the one endpoint the future HRMS attendance flow is expected to
call on every check-in/check-out attempt:

- If HRMS already knows which employee is attempting to check in (the common case — an employee
  badges in, or opens an app already logged in), it should supply `external_id` → **Mode B**
  (1:1 verification against that specific employee's enrolled embeddings).
- If the workflow is identity-agnostic (e.g. a shared kiosk where the employee hasn't identified
  themselves yet), HRMS omits `external_id` → **Mode A** (1:N identification against everyone
  enrolled), and the response's `external_id` tells HRMS who was recognized (only when the match
  clears the threshold).

## 3. Required request/response fields

**Request** (`multipart/form-data`):

| Field | Required | Notes |
|---|---|---|
| `file` | yes | One image/frame. |
| `external_id` | conditional | Required for Mode B (employee-specific verification); omit for Mode A. HRMS's employee ID is expected to be passed through unchanged as this opaque string — FaceVerification assigns it no meaning of its own. |

**Response** (`application/json`, `FaceVerificationResponse`):

| Field | Type | Meaning |
|---|---|---|
| `verified` | bool | Same information as `status`, as a boolean for easy branching. |
| `status` | `"PASS"` \| `"FAIL"` | The verdict. HRMS should treat this as the sole verdict. |
| `external_id` | string \| null | Echoed back in Mode B regardless of PASS/FAIL. In Mode A, populated only on PASS — never a low-confidence guess. |
| `similarity` | float | Raw cosine similarity in `[-1, 1]` (in practice usually `[0, 1]`) between the probe and the matched/claimed embedding. |
| `threshold` | float | The threshold that was actually applied for this decision (Mode B and Mode A use different configured thresholds — see `README.md` → "Configuration"). |
| `mode` | `"verification"` \| `"identification"` | Which mode this response came from. |
| `detection_score` | float | SCRFD's confidence in the detected face. |
| `quality` | object | `{accepted, quality_score, reasons, metrics}` — present on every response, informative even on PASS. |
| `processed_at` | datetime (ISO 8601, UTC) | For HRMS-side audit logs / correlation. |

**Error response** (non-2xx, `application/json`, `ErrorResponse`):

| Field | Type | Meaning |
|---|---|---|
| `error_code` | string | Stable, machine-checkable code (e.g. `no_face_detected`, `identity_not_found`, `model_not_ready`) — see `README.md` for the full list and HTTP status mapping. |
| `message` | string | Human-readable detail, safe to log but not guaranteed stable across versions — HRMS should branch on `error_code`, not on `message` text. |
| `request_id` | string \| null | Correlates to `X-Request-ID` — HRMS should log this alongside its own request/trace id. |
| `details` | object | Extra structured context (e.g. quality `reasons`, similarity/threshold on a rejected pair). |
| `timestamp` | datetime | When the error was generated. |

## 4. Authentication recommendation

The FaceVerification API already has an optional `X-API-Key` header check
(`API_KEY_ENABLED` / `API_KEY` in settings) — currently disabled by default for local development.
For a real HRMS integration:

- **Enable `API_KEY_ENABLED=true`** in every non-local environment. This is service-to-service
  traffic (HRMS backend → FaceVerification backend), not an end-user-facing API, so a static
  shared secret is proportionate — there is no need for a full OAuth2/end-user auth flow here.
- **Network-level isolation first, API key second.** The service should not be reachable from the
  public internet at all — place it on a private network/VPC/service mesh alongside HRMS, and
  treat the API key as defense-in-depth rather than the only control.
- **Secret management**: the API key should come from HRMS's existing secrets manager
  (whatever HRMS already uses for its own DB credentials etc.), not a hardcoded value — this is
  the one point where "inspect HRMS's existing secret-handling convention" (see checklist below)
  actually matters, so the same pattern is reused rather than a second, inconsistent one introduced.
- **Rotation**: support two valid keys during rotation windows if HRMS's secret manager doesn't
  already provide atomic rotation — this API's `require_api_key` dependency would need a small
  extension to accept a list of valid keys rather than one (flagged in section 8).
- mTLS is a reasonable upgrade if HRMS's infrastructure already terminates service-to-service TLS
  with client certs elsewhere — not required to start, but not in conflict with the above.

## 5. Timeout and error-handling recommendations

- **Timeout budget**: CPU-only ONNX inference (SCRFD + MobileFaceNet) on a single image is
  expected to complete well under a second once models are warm; recommend HRMS set an outbound
  HTTP timeout in the **3–5 second** range to absorb cold starts, GC pauses, and network jitter,
  with a short **connect timeout** (~1s) separate from the read timeout so a genuinely unreachable
  service fails fast rather than hanging the attendance flow.
- **Fail closed, not open.** If the call times out, errors, or the service is unreachable, HRMS
  must **not** treat that as PASS. A verification gate that silently passes on infrastructure
  failure defeats its own purpose. HRMS's existing error-handling flow (per the architecture
  diagram) should treat "could not verify" as its own outcome, distinct from "verified and failed".
- **Retry policy**: a single retry with a short backoff (e.g. 1 retry after ~500ms) is reasonable
  for transient network errors or a `503 model_not_ready` (which can occur if the service just
  restarted and hasn't finished loading models — see `/health/ready`). Do **not** retry on 4xx
  responses (`400`, `404`, `409`, `422`) — those are deterministic outcomes for the given input and
  will not change on retry.
- **Idempotency for enrollment**: `POST /api/v1/faces/enroll` has side effects (writes to FAISS).
  If HRMS's own retry logic could resubmit an enrollment request, be aware that a retried call
  after a genuine first-attempt success will hit `identity_already_exists` (409) under the default
  `reject` duplicate policy — HRMS should treat that 409 as "already done", not as a failure to
  surface to the end user.
- **Correlate request IDs**: forward the response's `request_id` (and/or HRMS's own trace id via
  a custom header, if HRMS's infra convention expects one) into HRMS's own logs, so a support
  investigation can jump straight from an HRMS log line to the matching FaceVerification log line.
- **Readiness check before routing real traffic**: HRMS's deployment/orchestration should probe
  `/health/ready` (not just `/health/live`) before considering the service available, since
  liveness returns 200 even before models finish loading.

## 6. What information must be returned to HRMS

At minimum, for every verification attempt HRMS needs:

1. **The verdict** (`status` / `verified`) — the only thing the existing attendance logic should
   branch on for "did this pass".
2. **Which identity was involved** (`external_id`) — to know whose attendance record to act on
   (Mode B: always; Mode A: only on PASS).
3. **The evidence behind the verdict** (`similarity`, `threshold`) — for audit trails, dispute
   resolution ("why was this rejected"), and for tuning thresholds later using real data.
4. **A reason when something is rejected before a verdict is even reached** (quality `reasons`,
   or the error `error_code` for detection/identity-not-found/model-not-ready failures) — so HRMS
   can show a meaningful message ("move closer to the camera", "try again in better lighting")
   instead of a generic failure, and so it can distinguish a face-quality problem from an actual
   non-match.
5. **A correlation id** (`request_id`) for support/debugging.

HRMS should **not** need, and this API will not provide, anything about attendance windows, shift
schedules, geofencing, or duplicate check-in prevention — those remain entirely HRMS's concern.

## 7. Checklist — existing HRMS files/modules to inspect later

**Not inspected as part of this task.** This is a category checklist to work from once an explicit
integration task begins; the actual file paths/names in the real HRMS codebase still need to be
located at that time.

- [ ] Existing attendance controller/service — where check-in/check-out is currently triggered,
      to find the insertion point for a FaceVerification call ahead of the existing logic.
- [ ] Existing employee/identity model or repository — to confirm what value HRMS should pass as
      `external_id` (employee code? internal numeric id? something else?) and that it's stable
      over an employee's lifetime (a value that changes on rehire/renaming would orphan
      enrollments).
- [ ] Existing outbound HTTP client conventions — whether HRMS already has a shared pattern for
      calling external services (retry/timeout wrapper, circuit breaker library, etc.) that this
      integration should reuse rather than duplicate.
- [ ] Existing secrets management — where API keys/credentials for other external integrations are
      stored, so the FaceVerification API key follows the same convention (see section 4).
- [ ] Existing error-handling/exception-mapping middleware — how HRMS currently surfaces
      third-party service failures to its own API consumers, to map FaceVerification's
      `error_code` values consistently with existing conventions.
- [ ] Existing logging/observability setup — log format, correlation id conventions, and whether
      there's a tracing system (e.g. OpenTelemetry) that a `request_id` should be attached to.
- [ ] Existing employee enrollment/onboarding workflow (UI and/or admin flow) — where the two
      enrollment images (`/faces/enroll`) would actually be captured/uploaded from.
- [ ] Existing employee offboarding workflow — where `DELETE /api/v1/enrollment/{external_id}`
      should be triggered to keep FAISS from retaining embeddings for departed employees.
- [ ] Existing infra/deployment config (docker-compose, k8s manifests, env/config files) — to
      determine how the FaceVerification service's base URL and API key will be provided to HRMS
      at runtime, and whether network policy needs updating to allow the connection.
- [ ] Existing rate-limiting/throttling layer, if any — to decide whether HRMS needs to protect
      this service from bursty attendance-time traffic (e.g. shift-change spikes).

## 8. Possible changes that may be required after the FaceVerification API is completed

These are informed guesses about what a real integration pass might surface — not commitments,
and none of them are implemented now:

- **Threshold tuning from production data.** `VERIFICATION_SIMILARITY_THRESHOLD` and
  `IDENTIFICATION_SIMILARITY_THRESHOLD` currently have reasonable defaults, but the right values
  depend on real false-accept/false-reject rates against HRMS's actual camera hardware and
  environments (lighting, kiosk placement, etc.) — expect these to be revisited after a pilot.
- **API key rotation support.** As noted in section 4, `require_api_key` currently checks against
  a single configured key; supporting rolling rotation may need it extended to accept a small set
  of currently-valid keys.
- **Multi-tenancy**, if this FaceVerification instance is meant to serve more than one HRMS
  deployment/organization — external_id alone may not be globally unique across tenants, which
  would require a tenant/organization scoping concept this API does not currently have.
- **Bulk/batch verification**, if HRMS's kiosk or attendance-capture workflow needs to process
  multiple frames or multiple people in one request rather than one image per call.
- **Webhook/async callback support**, if HRMS's capture flow turns out to need to submit a frame
  and be notified later rather than blocking on a synchronous response (e.g. for a queued,
  higher-latency capture pipeline).
- **Image pre-validation/virus scanning upstream**, if uploaded frames could originate from
  untrusted end-user devices rather than controlled kiosk hardware — this API validates that a
  file *decodes as an image*, not that it's safe to store/process from a security standpoint
  beyond that.
- **Per-caller rate limiting**, if shift-change traffic spikes turn out to need protection beyond
  what HRMS's own throttling (if any) provides.
- **Legacy embedding migration**, if HRMS (or a prior system) already has an existing face
  enrollment dataset — those embeddings would need re-enrollment through this API's own pipeline
  rather than direct import, since they'd have been produced by a different model/preprocessing.
- **Audit/reporting endpoint**, if HRMS wants historical verification attempts (not just live
  PASS/FAIL) surfaced from this side rather than reconstructed purely from HRMS's own logs.

None of the above should be assumed necessary — they are candidates to confirm or discard once the
actual HRMS codebase and pilot results are available.
