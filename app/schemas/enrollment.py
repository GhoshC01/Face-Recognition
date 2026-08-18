from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import QualityResultSchema


class EnrollmentResponse(BaseModel):
    external_id: str
    enrolled: bool
    embedding_id: int
    detection_score: float
    quality: QualityResultSchema
    message: str = "Face enrolled successfully"


class EnrollmentDeleteResponse(BaseModel):
    external_id: str
    removed: bool
    embeddings_removed: int


class EnrollmentStatusResponse(BaseModel):
    external_id: str
    enrolled: bool
    embedding_count: int


class EnrolledImageInfo(BaseModel):
    image: str
    embedding_id: int
    detection_score: float
    quality: QualityResultSchema


class DualImageEnrollmentResponse(BaseModel):
    """Response for the two-image initial enrollment workflow
    (POST /api/v1/faces/enroll)."""

    success: bool = True
    external_id: str
    images_processed: int
    enrollment_status: Literal["success"] = "success"
    image_similarity: float
    images: list[EnrolledImageInfo]
    message: str = "Face enrollment completed successfully"
