from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from app.config.settings import Settings
from app.core.detector import FaceDetector
from app.core.embedding import FaceEmbedder
from app.core.recognizer import FaceRecognizer
from app.core.vector_store import VectorStore
from app.services.enrollment_service import EnrollmentService
from app.services.evaluation_service import EvaluationService
from app.services.verification_service import VerificationService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_detector(request: Request) -> FaceDetector:
    return request.app.state.detector


def get_embedder(request: Request) -> FaceEmbedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_recognizer(request: Request) -> FaceRecognizer:
    return request.app.state.recognizer


def get_enrollment_service(request: Request) -> EnrollmentService:
    return request.app.state.enrollment_service


def get_verification_service(request: Request) -> VerificationService:
    return request.app.state.verification_service


def get_evaluation_service(request: Request) -> EvaluationService:
    return request.app.state.evaluation_service


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings: Settings = request.app.state.settings
    if not settings.api_key_enabled:
        return

    if x_api_key is None or x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
