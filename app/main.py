from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.rate_limiting import InMemoryRateLimiter
from app.api.routes import health as health_routes
from app.api.routes.v1 import api_router as api_router_v1
from app.api.timeout_middleware import RequestTimeoutMiddleware
from app.config.logging_config import configure_logging
from app.config.settings import get_settings
from app.core.detector import FaceDetector
from app.core.embedding import FaceEmbedder
from app.core.exceptions import ModelNotReadyError
from app.core.quality import QualityChecker, QualityThresholds
from app.core.recognizer import FaceRecognizer
from app.core.vector_store import VectorStore
from app.services.enrollment_service import EnrollmentService
from app.services.evaluation_service import EvaluationService
from app.services.verification_service import VerificationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    if settings.api_key_enabled and settings.api_key == "changeme":
        if settings.environment == "production":
            raise RuntimeError(
                "API_KEY_ENABLED is true but API_KEY is still the default placeholder "
                "'changeme'. Refusing to start in production -- set a real secret."
            )
        logger.warning("api_key_is_default_placeholder", extra={"environment": settings.environment})

    detector = FaceDetector(
        model_path=settings.detector_model_path,
        input_size=settings.detector_input_size,
        confidence_threshold=settings.detector_confidence_threshold,
        nms_threshold=settings.detector_nms_threshold,
        intra_op_threads=settings.onnx_intra_op_threads,
        inter_op_threads=settings.onnx_inter_op_threads,
    )
    embedder = FaceEmbedder(
        model_path=settings.recognizer_model_path,
        input_size=settings.recognizer_input_size,
        intra_op_threads=settings.onnx_intra_op_threads,
        inter_op_threads=settings.onnx_inter_op_threads,
    )

    try:
        detector.load()
        embedder.load()
        if embedder.embedding_dimension != settings.embedding_dimension:
            logger.warning(
                "embedding_dimension_mismatch",
                extra={
                    "model_reported_dimension": embedder.embedding_dimension,
                    "configured_dimension": settings.embedding_dimension,
                },
            )
    except (FileNotFoundError, ModelNotReadyError) as exc:
        logger.warning(
            "model_load_failed_at_startup",
            extra={"reason": str(exc)},
        )

    quality_checker = QualityChecker(
        QualityThresholds(
            min_detection_confidence=settings.quality_min_detection_confidence,
            min_face_width_px=settings.quality_min_face_width_px,
            min_face_height_px=settings.quality_min_face_height_px,
            min_face_area_ratio=settings.quality_min_face_area_ratio,
            min_brightness=settings.quality_min_brightness,
            max_brightness=settings.quality_max_brightness,
            min_sharpness=settings.quality_min_sharpness,
        )
    )
    recognizer = FaceRecognizer(
        detector=detector,
        embedder=embedder,
        quality_checker=quality_checker,
        max_faces_to_consider=settings.max_faces_to_consider,
    )
    vector_store = VectorStore(
        dimension=settings.embedding_dimension,
        index_dir=settings.faiss_index_dir,
        metadata_dir=settings.metadata_dir,
        index_filename=settings.faiss_index_filename,
        metadata_filename=settings.metadata_filename,
    )

    app.state.settings = settings
    app.state.detector = detector
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.recognizer = recognizer
    app.state.enrollment_service = EnrollmentService(
        recognizer=recognizer,
        vector_store=vector_store,
        duplicate_policy=settings.enrollment_duplicate_policy,
        images_dir=settings.enrollment_images_dir,
    )
    app.state.verification_service = VerificationService(
        recognizer=recognizer,
        vector_store=vector_store,
        verification_threshold=settings.verification_similarity_threshold,
        identification_threshold=settings.identification_similarity_threshold,
        identification_top_k=settings.identification_top_k,
        multi_frame_min_frames=settings.multi_frame_min_frames,
        multi_frame_max_frames=settings.multi_frame_max_frames,
        multi_frame_min_valid_frames=settings.multi_frame_min_valid_frames,
        multi_frame_min_agreeing_frames=settings.multi_frame_min_agreeing_frames,
        multi_frame_consensus_ratio=settings.multi_frame_consensus_ratio,
    )
    app.state.evaluation_service = EvaluationService(
        recognizer=recognizer,
        similarity_threshold=settings.verification_similarity_threshold,
    )

    logger.info("service_started", extra={"environment": settings.environment})
    yield
    logger.info("service_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Reusable face detection, alignment, embedding, and verification service. "
            "Stateless with respect to any caller's business domain: it stores only "
            "opaque external_id -> face-embedding mappings."
        ),
        lifespan=lifespan,
    )

    # Middleware is added innermost-first: Starlette makes whichever
    # middleware is added LAST the OUTERMOST layer. This ordering results in
    # the following execution order for an incoming request (outermost to
    # innermost): HTTPS redirect -> CORS -> request context (request id +
    # access log) -> request timeout -> rate limit -> routing.
    if settings.rate_limit_enabled:
        app.add_middleware(
            InMemoryRateLimiter,
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=settings.request_timeout_seconds)
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_allow_origins:
        # A "*" origin can never be combined with credentials -- browsers
        # reject that combination outright, and allowing it would be a
        # misconfiguration risk even where a browser happened to tolerate it.
        allow_credentials = "*" not in settings.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if settings.enforce_https:
        app.add_middleware(HTTPSRedirectMiddleware)

    register_exception_handlers(app)

    app.include_router(health_routes.router)
    app.include_router(api_router_v1, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["root"])
    def root() -> dict[str, str]:
        return {"service": settings.app_name, "version": settings.app_version, "status": "running"}

    return app


app = create_app()
