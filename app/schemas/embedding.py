from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.schemas.common import QualityResultSchema


class EmbeddedImageSchema(BaseModel):
    index: int
    embedding: list[float]
    detection_score: float
    quality: QualityResultSchema
    similarity_to_first: float | None = None


class EmbedResponse(BaseModel):
    """Response for POST /api/v1/faces/embed.

    Stateless embedding extraction: no enrollment or vector-store write.
    A single image returns one embedding. Multiple images are accepted only
    when every later image matches the first (cosine similarity >=
    `consistency_threshold`); otherwise the request fails and no embeddings
    are returned.
    """

    images_processed: int
    embeddings: list[EmbeddedImageSchema]
    consistency_checked: bool
    consistency_threshold: float | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
