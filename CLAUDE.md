# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## এই প্রজেক্ট আসলে কী

**Face Verification API** — একটা reusable, ONNX-based face detection/recognition/verification service। এটা কোনো নির্দিষ্ট business domain (যেমন HRMS/attendance) সম্পর্কে কিছুই জানে না। এটা শুধু `external_id -> face embedding` mapping রাখে, আর একটা request-এর জন্য similarity-based **PASS/FAIL** result ফেরত দেয়। কে "external_id" (employee code, visitor id, customer id, যা খুশি) — সেটা caller (যেমন HRMS) ঠিক করে।

```
Client App -> HRMS API -> এই Face Verification API -> Detection -> Alignment -> Embedding
                                                     -> FAISS -> Verification Result
                        -> HRMS API -> PASS/FAIL -> Attendance/Error (HRMS handle করে)
```

**এই service যা করে না:** attendance logic, employee business logic (leave/shift/payroll), HRMS-এর কোনো ডেটা টাচ করা। `external_id` একটা opaque string মাত্র।

---

## Commands (কমন কমান্ডসমূহ)

### প্রথমবার সেটআপ
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # Linux/macOS

pip install -r requirements-dev.txt
copy .env.example .env
```
⚠️ **সবসময় নিশ্চিত করুন venv আসলেই active** (`python -c "import sys; print(sys.executable)"` — path টা `.venv\Scripts\python.exe` দেখাতে হবে, `AppData\Roaming\Python\...` না)। venv activate না থাকলে dependencies global-এ install হয়ে যায়, আর তখন `str | None` টাইপ পার্স করতে গিয়ে pydantic crash করে (Python 3.9 + `eval_type_backport` missing থাকলে)।

### Server চালানো
```powershell
uvicorn app.main:app --reload --port 8000
```
- `det_500m.onnx` আর `w600k_mbf.onnx` — এই দুটো ONNX model file [models/](models/) ফোল্ডারে না থাকলেও server চালু হবে (`/health/live` 200 দিবে), কিন্তু `/health/ready` **503** দিবে যতক্ষণ না দুটো model load হয়।
- Model file না থাকলে সংগ্রহ করার উপায়:
  ```powershell
  pip install insightface
  python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_sc')"
  ```
  এটা `~/.insightface/models/buffalo_sc/`-এ ডাউনলোড করবে — সেখান থেকে `det_500m.onnx` আর `w600k_mbf.onnx` কপি করে `models/`-এ আনুন।
- Interactive docs: `http://localhost:8000/docs`

### Test চালানো
```powershell
pytest                              # সব টেস্ট
pytest tests/test_quality.py        # একটামাত্র ফাইল
pytest tests/test_quality.py::test_rejects_dark_image   # একটামাত্র টেস্ট
pytest -k "enrollment"              # নাম-ম্যাচ দিয়ে
```
টেস্ট চালাতে real ONNX model লাগে না — `tests/conftest.py`-এর `FakeRecognizer` marker-byte দিয়ে deterministic fake embedding বানায়, তাই service/route লেভেলের টেস্ট real ONNX Runtime ছাড়াই চলে।

### Docker
```bash
export FACE_API_KEY="local-testing-key-change-me"   # docker-compose.yml-এ এটা required
docker compose up --build
```

### Offline accuracy benchmark (endpoint না, CLI tool)
```bash
python scripts/run_benchmark.py --dataset path/to/manifest.json
```

### Stale enrollment purge (retention/minimization tool)
```bash
python scripts/purge_stale_enrollments.py --older-than-days 365            # শুধু লিস্ট করে
python scripts/purge_stale_enrollments.py --older-than-days 365 --delete   # সত্যিই মুছে দেয়
```

---

## High-level architecture (স্তরে স্তরে)

```
app/
├── api/            <- HTTP layer: routes, auth, middleware, error translation
├── core/           <- বিশুদ্ধ recognition pipeline (কোনো HTTP/FAISS জ্ঞান নেই এদের নিজেদের ভেতর আলাদা করে)
├── services/        <- use-case orchestration: core + vector_store একসাথে জুড়ে দেয়
├── evaluation/       <- অফলাইন accuracy measurement (production থেকে আলাদা)
├── schemas/           <- pydantic request/response contracts
├── config/             <- env-driven settings + logging setup
├── utils/               <- image decode, request-id context
└── main.py               <- app factory + lifespan (model loading) + router wiring
```

**গুরুত্বপূর্ণ layering rule:** `app/core/*` কখনো HTTP request, FastAPI, বা external caller সম্পর্কে জানে না — শুধু numpy/cv2/onnxruntime/faiss নিয়ে কাজ করে। `app/services/*` হলো middle layer যেটা `core` (recognizer + vector_store) একসাথে জুড়ে ব্যবসায়িক use-case (enroll/verify/identify) বানায়। `app/api/*` শুধু HTTP-specific জিনিস (upload validation, auth, error JSON shape) হ্যান্ডেল করে, তারপর কাজ `services`-কে দিয়ে দেয়। এই আলাদা করাটা `docs/final-architecture-review.md`-তে import-graph দিয়ে verify করা হয়েছে।

---

## ফাইল-ভিত্তিক বিস্তারিত (কোন ফাইল কী কাজ করে)

### `app/main.py`
App factory + `lifespan` context manager। Server startup-এ যা হয়:
1. Settings লোড, logging কনফিগার।
2. `FaceDetector` আর `FaceEmbedder` তৈরি করে `.load()` কল করে — model file না পেলে **crash করে না**, শুধু warning log করে (`model_load_failed_at_startup`), যাতে `/health/live` তখনও কাজ করে।
3. `QualityChecker`, `FaceRecognizer` (detector+embedder+quality একসাথে), `VectorStore` তৈরি।
4. `EnrollmentService`, `VerificationService`, `EvaluationService` — সবকিছু `app.state`-এ রাখে যাতে routes dependency injection দিয়ে অ্যাক্সেস করতে পারে।
5. Middleware ordering (গুরুত্বপূর্ণ — শেষে add করা middleware হয় সবচেয়ে বাইরের layer): HTTPS redirect -> CORS -> request context (id+access log) -> timeout -> rate limit -> routing।
6. `/` root route — সার্ভিসের নাম/ভার্সন/status রিটার্ন করে (আপনি browser-এ এটাই দেখেছিলেন)।

### `app/config/settings.py`
`pydantic-settings`-ভিত্তিক single `Settings` class — সব env var এখানে টাইপ+ডিফল্ট সহ ডিফাইন করা। `.env` ফাইল থেকে অটো-লোড হয় (`env_file=".env"`)। প্রতিটা group-এর জন্য comment block আছে (Application, Security, Models, Detection, Recognition, Enrollment, Multi-frame, Quality gates, Storage, Uploads)। `enrollment_images_dir` (ডিফল্ট `"images"`) — সবচেয়ে নতুন addition, নিচে "Enrollment photo saving" সেকশনে বিস্তারিত।

### `app/config/logging_config.py`
`JsonLogFormatter` — প্রতিটা log line-কে single-line JSON বানায় (timestamp, level, logger, message, **request_id** — automatic contextvar থেকে, `extra={}` না দিলেও)। `configure_logging()` root handler clear করে duplicate log আটকায়, আর `uvicorn.access`-কে WARNING-এ silence করে (কারণ নিজস্ব `RequestContextMiddleware` আগে থেকেই access log করে)।

### `app/api/deps.py`
FastAPI dependency provider ফাংশনগুলো — সব `request.app.state` থেকে জিনিস বের করে দেয় (`get_settings`, `get_detector`, `get_recognizer`, `get_enrollment_service`, ইত্যাদি)। `require_api_key` — `API_KEY_ENABLED=true` হলে `X-API-Key` header চেক করে, নাহলে no-op।

### `app/api/middleware.py`
`RequestContextMiddleware` — প্রতি request-এ একটা request-id বসায় (`X-Request-ID` header থেকে reuse করে অথবা নতুন বানায়), duration measure করে, একটা structured access-log line লেখে, response-এ `X-Request-ID` echo করে।

### `app/api/exception_handlers.py`
সব domain error (`app/core/exceptions.py`-এর) কে uniform JSON shape-এ কনভার্ট করে:
```json
{"error_code": "...", "message": "...", "request_id": "...", "details": {}, "timestamp": "..."}
```
Unhandled exception পুরো traceback সহ server-side log হয়, কিন্তু client কখনো internal details পায় না — generic 500 ফেরত যায়। এটাই "internals never leak" invariant।

### `app/api/upload_validation.py`
Image decode করার **আগে** HTTP-লেভেলে চেক করে: content-type allow-list (`ALLOWED_CONTENT_TYPES`), size limit (`MAX_UPLOAD_SIZE_MB`) — declared size আর real byte count দুটোই চেক হয়।

### `app/api/rate_limiting.py`
`InMemoryRateLimiter` — sliding-window, per-process, in-memory। API key বা IP দিয়ে key করে। **সীমাবদ্ধতা**: multi-replica deployment-এ প্রতিটা replica নিজের counter রাখে, তাই global limit ঠিকমতো enforce হয় না — gateway-level limiting-এর replacement না, শুধু defense-in-depth।

### `app/api/timeout_middleware.py`
`RequestTimeoutMiddleware` — `REQUEST_TIMEOUT_SECONDS` (ডিফল্ট 30s) পার হলে 504 রিটার্ন করে, যাতে একটা slow ONNX call worker আটকে না রাখে।

### `app/api/routes/health.py`
`/health/live` (সবসময় 200, process বেঁচে আছে কিনা) আর `/health/ready` (detector+embedder loaded + vector store ready — নাহলে 503)। Docker `HEALTHCHECK` আর k8s readiness probe এটাই ব্যবহার করে।

### `app/api/routes/v1/faces.py` — **মূল ইউজার-ফেসিং endpoints**
- `POST /faces/enroll` — **প্রাথমিক enrollment, দুটো ছবি বাধ্যতামূলক** (`image1`, `image2`)। দুটোতেই exactly একটা face থাকতে হবে, quality gate পাস করতে হবে, আর দুটোর মধ্যে cosine similarity ≥ `ENROLLMENT_MIN_IMAGE_SIMILARITY` (একই মানুষ কিনা চেক)।
- `POST /faces/verify` — HRMS-এর main endpoint। `external_id` দিলে Mode B (1:1 verify), না দিলে Mode A (1:N identify)। **exactly একটা face লাগবে** (একাধিক face পেলে 422)।
- `POST /faces/verify-multi` — webcam multi-frame consensus verification (নিচে flow সেকশনে বিস্তারিত)।

### `app/api/routes/v1/enrollment.py` — **সেকেন্ডারি enrollment API**
- `POST /enrollment` (form: `external_id`, `file`) — **একটামাত্র ছবি** দিয়ে enroll/add-more-embedding। ইতিমধ্যে enrolled কারো নতুন angle যোগ করতে ব্যবহার হয়।
- `GET /enrollment/{external_id}` — enrolled কিনা, কয়টা embedding আছে।
- `DELETE /enrollment/{external_id}` — সব embedding মুছে দেয়, **আর সেই identity-র saved photo folder-ও মুছে দেয়** (নতুন behavior, নিচে দেখুন)।

### `app/api/routes/v1/verification.py` — **"Legacy"/generic primitives**
- `POST /verification/verify` — 1:1, PASS/FAIL।
- `POST /verification/identify` — 1:N, `top_k` matches লিস্ট রিটার্ন করে।
- `POST /verification/compare` — দুটো ছবি stateless compare, কোনো enrollment/storage ছাড়াই।

`faces.py`-এর endpoint গুলো `strict_single_face=True` ব্যবহার করে (ambiguous frame কখনো silently resolve করা যাবে না, যেহেতু এটাই primary attendance-capture path), আর `verification.py`-এর গুলো `strict_single_face=False` (একাধিক face পেলে সবচেয়ে বড়টা বেছে নেয়) — এটা ইচ্ছাকৃত, ডকুমেন্টেড পার্থক্য।

### `app/core/detector.py` — SCRFD face detector (ONNX wrapper)
`FaceDetector.detect(image) -> list[Face]`। প্রতিটা `Face`-এ box (x1,y1,x2,y2), confidence score, আর 5-point landmarks। Letterbox-resize + custom NMS দিয়ে raw ONNX আউটপুট থেকে face box বের করে। Model file miss করলে `ModelNotReadyError`।

### `app/core/alignment.py`
`align_face(image, landmarks, output_size=112)` — 5-point landmark দিয়ে ArcFace-এর canonical template-এর সাথে similarity-transform মিলিয়ে face crop-কে সোজা (frontal) করে। Degenerate landmarks (সব পয়েন্ট এক জায়গায়) হলে `InvalidImageError`।

### `app/core/quality.py`
`QualityChecker.evaluate(image, box, detection_confidence) -> QualityResult`। brightness, sharpness (Laplacian variance), face size (pixel + area ratio), detection confidence — সবগুলো চেক **সম্পূর্ণ রান হয়** (একটা fail করলেই থেমে যায় না), তাই একসাথে একাধিক reason (`face_too_small`, `image_too_blurry` ইত্যাদি) আসতে পারে। `quality_score` কোনো accuracy measure না, শুধু image-quality signal।

### `app/core/embedding.py` — MobileFaceNet/ArcFace embedder
`FaceEmbedder.embed(aligned_face) -> np.ndarray`। Embedding dimension **model থেকে নিজে থেকে read করে** (কখনো hardcode না) — mismatch হলে warning/error। আউটপুট সবসময় `l2_normalize` দিয়ে পাস করা।

### `app/core/normalization.py`
`l2_normalize(embedding)` — vector-কে unit length-এ scale করে, যাতে cosine similarity = plain dot product হয় (FAISS `IndexFlatIP`-এর জন্য দরকারি)। `None`/wrong-shape/NaN/near-zero — সবকিছুতে `InvalidEmbeddingError`।

### `app/core/recognizer.py` — **পুরো pipeline-এর orchestrator**
`FaceRecognizer.process(image, strict_single_face) -> FaceEmbeddingResult`। এটাই সব service (`EnrollmentService`, `VerificationService`, `EvaluationService`, benchmark) ব্যবহার করে। ভেতরে: detect -> face select (single/largest) -> quality gate -> align -> embed -> normalize। কোনো face না পেলে `NoFaceDetectedError`, একাধিক পেলে (strict মোডে) `MultipleFacesDetectedError`, quality fail করলে `LowImageQualityError`।

### `app/core/vector_store.py` — FAISS + JSON metadata
`VectorStore` — `IndexIDMap2(IndexFlatIP(dimension))` ব্যবহার করে (normalized vector-এ inner product = cosine similarity)। দুটো ফাইলে persist হয়: FAISS index (`storage/faiss/index.faiss`) + JSON metadata sidecar (`storage/metadata/metadata.json` — `external_id <-> faiss_id` mapping + enrollment timestamp)। **কোনো ছবি বা raw data এখানে নেই, শুধু vector + mapping।** Atomic save (temp file + rename), corrupted/mismatched ফাইল পেলে reset করে empty store-এ (crash করে না)। মূল মেথড: `add_embedding`, `search` (top-k, প্রতি identity-র best score নেয়), `remove_embedding`, `get_embeddings`, `list_identities`, `get_last_enrolled_at`।

### `app/core/exceptions.py`
সব domain error-এর কেন্দ্রীয় hierarchy — প্রতিটার নিজস্ব `error_code` আর `http_status`। যেমন: `NoFaceDetectedError` (422), `LowImageQualityError` (422), `IdentityNotFoundError` (404), `IdentityAlreadyExistsError` (409), `ModelNotReadyError` (503)।

### `app/services/enrollment_service.py`
`EnrollmentService` — `enroll()` (single image, add embedding) আর `enroll_pair()` (দুই-ছবি প্রাথমিক enrollment, cross-validate করে দুটো একই মানুষ কিনা)। `duplicate_policy` (`reject`/`replace`) আর rollback logic (দ্বিতীয় FAISS write fail করলে প্রথমটাও রোলব্যাক)। **নতুন**: `images_dir` parameter — সেট করা থাকলে প্রতিটা accepted enrollment ছবি ডিস্কে সেভ করে (নিচে বিস্তারিত)।

### `app/services/verification_service.py`
`VerificationService` — `verify_or_identify()` (প্রধান `/faces/verify`-এর লজিক, strict single-face), `verify_multi_frame()` (multi-frame consensus, নিচে flow সেকশনে), `verify()`/`identify()` (legacy endpoints, lenient face selection)।

### `app/services/evaluation_service.py`
`EvaluationService.compare(image_a, image_b)` — stateless, কোনো FAISS/enrollment ছাড়াই দুটো ছবির embedding সরাসরি compare করে।

### `app/evaluation/*` — অফলাইন accuracy measurement (production থেকে সম্পূর্ণ আলাদা)
- `dataset.py` — labeled gallery+probe JSON manifest লোড করে।
- `metrics.py` — accuracy/precision/recall/FAR/FRR/confusion-matrix/threshold-sweep গণনা করে।
- `benchmark.py` — `BenchmarkRunner`, আসল production pipeline (`FaceRecognizer`) ব্যবহার করে কিন্তু **নিজস্ব temporary FAISS store**-এ (production ডেটা টাচ করে না, যদি না `--use-production` দেওয়া হয়, তখনও শুধু read-only `search()`)।

### `app/schemas/*`
সব API request/response contract এখানে pydantic model হিসেবে — `common.py` (error envelope + quality schema), `enrollment.py`, `verification.py`, `health.py`।

### `app/utils/image_utils.py`
`decode_image_bytes(bytes) -> np.ndarray` (cv2.imdecode দিয়ে BGR ndarray বানায়), `crop_box()`।

### `app/utils/request_context.py`
Contextvar-ভিত্তিক request-id — middleware, error handler, আর logging formatter — সবাই এটা শেয়ার করে।

### `tests/`
- `conftest.py` — `FakeRecognizer` (marker byte দিয়ে deterministic fake embedding, real ONNX ছাড়াই টেস্ট চালানো যায়), `make_synthetic_image()`, `client` fixture (isolated tmp storage)।
- বাকি সব `test_*.py` — প্রতিটা module/route-এর জন্য আলাদা টেস্ট ফাইল।

### `scripts/`
- `run_benchmark.py` — offline accuracy benchmark CLI।
- `purge_stale_enrollments.py` — retention/minimization tool, পুরনো enrollment লিস্ট/মুছে দেয় (এখন ছবির folder-ও মুছে দেয়)।

### Docker ফাইল
- `Dockerfile` — two-stage (builder + runtime), non-root user, models/storage কখনো bake-in হয় না।
- `docker-compose.yml` — local single-instance, `FACE_API_KEY` required, models read-only mount, storage named volume।
- `docker/entrypoint.sh` — production ASGI startup (`uvicorn` exec, graceful shutdown)।

### Docs
- `README.md` — পূর্ণাঙ্গ API reference + উদাহরণ।
- `SECURITY.md` — auth, upload validation, CORS, rate limiting, logging invariant, biometric data minimization।
- `docs/deployment.md` — Docker/production deployment গাইড।
- `docs/final-architecture-review.md` — evidence-based production-readiness review।
- `docs/hrms-integration-plan.md` — ভবিষ্যতের HRMS integration পরিকল্পনা (এখনো implement হয়নি)।

---

## Step-by-step flow (পুরো request pipeline)

### 1) Enrollment flow — দুই-ছবি প্রাথমিক enrollment (`POST /api/v1/faces/enroll`)

```
1. Client পাঠায়: external_id, image1, image2 (multipart form)
2. app/api/upload_validation.py -> content-type + size চেক (দুটো ছবির জন্যই আলাদা করে)
3. app/utils/image_utils.py -> decode_image_bytes() -> numpy array (দুটো ছবিই)
4. app/services/enrollment_service.py :: enroll_pair()
   ক) duplicate_policy="reject" আর আগে থেকে enrolled থাকলে -> IdentityAlreadyExistsError (409), এখানেই থেমে যায়
   খ) image1 -> FaceRecognizer.process(strict_single_face=True)
        -> detector.detect() -> face select (exactly ১টা face লাগবে, নাহলে error)
        -> quality.evaluate() -> fail করলে LowImageQualityError
        -> align_face() -> embedder.embed() -> l2_normalize()
        = embedding1
   গ) image2 -> ঠিক একই ধাপ = embedding2
   ঘ) image_similarity = dot(embedding1, embedding2)
      এটা ENROLLMENT_MIN_IMAGE_SIMILARITY (ডিফল্ট 0.40) থেকে কম হলে
      -> InconsistentEnrollmentImagesError (422) - "দুটো ছবি একই মানুষের মনে হচ্ছে না"
   ঙ) duplicate_policy="replace" আর already enrolled হলে -> পুরনো embedding মুছে ফেলা হয় (এই পয়েন্টেই, আগে না)
   চ) vector_store.add_embedding(external_id, embedding1) -> embedding_id_1
   ছ) vector_store.add_embedding(external_id, embedding2) -> embedding_id_2
      (দ্বিতীয়টা fail করলে প্রথমটাও রোলব্যাক হয় - partial enrollment কখনো থাকে না)
   জ) [নতুন] দুটো ছবিই images/{external_id}/{embedding_id}.jpg হিসেবে ডিস্কে সেভ হয় (best-effort)
5. Response: success, external_id, images_processed=2, image_similarity, প্রতিটা ছবির detection_score+quality
```

### 2) Enrollment flow — একটামাত্র ছবি (`POST /api/v1/enrollment`)
একই pipeline (ধাপ 2-4-খ পর্যন্ত), শুধু একটা ছবি, cross-validation নেই। ইতিমধ্যে enrolled কারো নতুন angle/embedding যোগ করার জন্য।

### 3) Verification/Identification flow (`POST /api/v1/faces/verify`)

```
1. Client পাঠায়: file, (optional) external_id
2. upload validation -> decode -> numpy array
3. FaceRecognizer.process(strict_single_face=True) -> embedding (একই ৪-ধাপ pipeline: detect->quality->align->embed->normalize)
4. VerificationService.verify_or_identify():
   Mode B (external_id দেওয়া আছে):
     - vector_store.get_embeddings(external_id) -> যদি কিছু enrolled না থাকে -> IdentityNotFoundError (404)
     - সেই identity-র সব embedding-এর সাথে dot product -> সর্বোচ্চ similarity
     - similarity >= VERIFICATION_SIMILARITY_THRESHOLD (ডিফল্ট 0.36) -> PASS, নাহলে FAIL
     - external_id সবসময় echo হয় (caller নিজেই দিয়েছে)
   Mode A (external_id নেই):
     - vector_store.search(embedding, top_k=1) -> সব enrolled identity-র মধ্যে সেরা match
     - similarity >= IDENTIFICATION_SIMILARITY_THRESHOLD (ডিফল্ট 0.40) -> PASS + external_id
     - না মিললে FAIL + external_id: null (কম-confidence guess কখনো surface হয় না)
5. Response: verified, status (PASS/FAIL), external_id, similarity, threshold, mode, detection_score, quality
```

### 4) Multi-frame consensus verification (`POST /api/v1/faces/verify-multi`)

```
1. Client পাঠায় কয়েকটা frame (৩-৫টা, webcam capture)
2. প্রতিটা frame আলাদাভাবে ৩নং flow-এর মতোই প্রসেস হয় (strict_single_face=True)
   - কোনো frame decode fail / no-face / multiple-face / low-quality হলে
     সেই frame শুধু বাদ পড়ে (rejection_reason সহ) - পুরো request fail হয় না
3. MULTI_FRAME_MIN_VALID_FRAMES (ডিফল্ট ২)-র কম valid frame থাকলে
   -> FAIL, reason: insufficient_valid_frames
4. যে identity সবচেয়ে বেশি frame-এ top-match হয়েছে, সেটাই "leading candidate"
5. frames_agreeing = leading candidate-এর মধ্যে কতগুলো frame individually threshold পার করেছে
   consensus_ratio = frames_agreeing / valid_frames
6. PASS হবে শুধু যদি:
   frames_agreeing >= MULTI_FRAME_MIN_AGREEING_FRAMES (ডিফল্ট ২)  AND
   consensus_ratio >= MULTI_FRAME_CONSENSUS_RATIO (ডিফল্ট 0.6)
   (একটামাত্র শক্তিশালী frame কখনো একাই PASS আনতে পারবে না)
7. debug=true দিলে response-এ প্রতিটা frame-এর diagnostic থাকবে
```

### 5) Enrollment removal flow (`DELETE /api/v1/enrollment/{external_id}`)
```
1. vector_store.has_identity() চেক - না থাকলে IdentityNotFoundError (404)
2. vector_store.remove_embedding() - সব embedding + metadata entry মুছে যায়
3. [নতুন] images/{external_id}/ পুরো ফোল্ডার মুছে যায় (best-effort)
```

---

## Enrollment photo saving (নতুন feature — গুরুত্বপূর্ণ)

**পটভূমি:** এই সার্ভিস আগে ছিল embeddings-only — raw ছবি কখনো disk-এ সেভ হতো না (in-memory pipeline: decode -> process -> discard)। এখন `ENROLLMENT_IMAGES_DIR` (ডিফল্ট `"images"`) সেট করা থাকলে প্রতিটা **accepted** enrollment ছবি সেভ হয়:

```
images/<external_id>/<embedding_id>.jpg
```

- `embedding_id` হলো `VectorStore.add_embedding()`-এর রিটার্ন করা globally-unique FAISS id, তাই একই external_id-তে একাধিকবার enroll করলেও filename কখনো collide করে না।
- Save করা হয় **শুধু FAISS write সফল হওয়ার পরে** (`EnrollmentService._save_image()`), never before — যাতে ব্যর্থ enrollment-এর জন্য এতিম (orphan) ছবি তৈরি না হয়।
- **Best-effort, non-blocking**: disk full/permission error হলে শুধু warning log হয় (`enrollment_image_save_failed`), enrollment response-এ কোনো প্রভাব পড়ে না — কারণ FAISS-ই সত্যিকারের source of truth, এই ছবি শুধু audit/debug copy।
- `EnrollmentService.__init__`-এ `images_dir` **default `None`** — মানে ফিচারটা opt-in, existing unit test-গুলো এই parameter না দিলে আগের মতো কোনো ফাইল লেখে না। শুধু `app/main.py` real app চালানোর সময় `settings.enrollment_images_dir` থেকে wire করে।
- বন্ধ করতে চাইলে: `.env`-এ `ENROLLMENT_IMAGES_DIR=` (খালি) সেট করুন।
- **cleanup সবসময় sync করা আছে**: `DELETE /api/v1/enrollment/{id}` আর `scripts/purge_stale_enrollments.py --delete` — দুটোই identity মুছলে তার photo folder-ও মুছে দেয়।
- `.gitignore`-এ `images/*` (except `.gitkeep`) ইগনোর করা আছে — real ছবি কখনো git-এ কমিট হবে না।
- **নথিভুক্ত trade-off**: `SECURITY.md`-এর "Biometric data minimization" সেকশন এই behavior প্রতিফলিত করার জন্য আপডেট করা হয়েছে (আগে লেখা ছিল "raw image কখনো retain হয় না" — এখন সেটা আর সত্যি না, তাই ডকুমেন্টেশন mismatch এড়াতে আপডেট করা হয়েছে)।

---

## Testing patterns মনে রাখার মতো

- **Real ONNX model লাগে না**: `tests/conftest.py`-এর `FakeRecognizer` একটা marker byte (`image[0,0,0]`) পড়ে deterministic embedding বানায় (seeded RNG) — একই marker সবসময় একই "মানুষ" simulate করে।
- **Isolated storage**: `client` fixture প্রতিটা টেস্টের জন্য `tmp_path`-এ আলাদা FAISS/metadata dir বসায়, তাই টেস্ট একে অপরের ডেটা নষ্ট করে না।
- Model path ইচ্ছাকৃতভাবে nonexistent path-এ সেট করা থাকে টেস্টে, যাতে `model_not_ready` behavior টেস্ট করা যায় real model ছাড়াই।
