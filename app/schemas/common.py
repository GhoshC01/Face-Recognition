from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.core.quality import QualityResult


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QualityMetricsSchema(BaseModel):
    detection_confidence: float
    face_width: int
    face_height: int
    face_area_ratio: float
    brightness: float
    sharpness: float


class QualityResultSchema(BaseModel):
    accepted: bool
    quality_score: float
    reasons: list[str] = Field(default_factory=list)
    metrics: QualityMetricsSchema | None = None


def quality_result_to_schema(result: QualityResult) -> QualityResultSchema:
    metrics = (
        QualityMetricsSchema(
            detection_confidence=result.metrics.detection_confidence,
            face_width=result.metrics.face_width,
            face_height=result.metrics.face_height,
            face_area_ratio=result.metrics.face_area_ratio,
            brightness=result.metrics.brightness,
            sharpness=result.metrics.sharpness,
        )
        if result.metrics is not None
        else None
    )
    return QualityResultSchema(
        accepted=result.accepted,
        quality_score=result.quality_score,
        reasons=result.reasons,
        metrics=metrics,
    )
