from __future__ import annotations

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
