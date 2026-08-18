from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.schemas.health import LivenessResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=LivenessResponse)
def liveness() -> LivenessResponse:
    """Process-is-running check. Always returns 200 if the app can respond at all."""
    return LivenessResponse()


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Dependency-aware check: models loaded and vector store reachable."""
    state = request.app.state
    detector_loaded = state.detector.is_loaded
    recognizer_loaded = state.embedder.is_loaded
    vector_store_ready = state.vector_store is not None

    is_ready = detector_loaded and recognizer_loaded and vector_store_ready
    response.status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        detector_loaded=detector_loaded,
        recognizer_loaded=recognizer_loaded,
        vector_store_ready=vector_store_ready,
        enrolled_identities=state.vector_store.count() if vector_store_ready else 0,
        app_version=state.settings.app_version,
    )
