from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import QualityResultSchema


class VerificationResponse(BaseModel):
    external_id: str
    verified: bool
    result: Literal["PASS", "FAIL"]
    similarity_score: float
    threshold: float
    detection_score: float
    quality: QualityResultSchema
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IdentificationMatch(BaseModel):
    external_id: str
    similarity_score: float


class IdentificationResponse(BaseModel):
    matches: list[IdentificationMatch]
    threshold: float
    detection_score: float
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompareResponse(BaseModel):
    match: bool
    result: Literal["PASS", "FAIL"]
    similarity_score: float
    threshold: float
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FaceCompareImageSchema(BaseModel):
    detection_score: float
    quality: QualityResultSchema


class FaceCompareResponse(BaseModel):
    """Response for POST /api/v1/faces/compare.

    Stateless two-image comparison: no enrollment or vector-store lookup.
    `confidence` is cosine similarity in [0, 1]; `status` is Match when
    confidence clears `threshold`, otherwise Not matching.
    """

    matched: bool
    status: Literal["Match", "Not matching"]
    confidence: float
    threshold: float
    image1: FaceCompareImageSchema
    image2: FaceCompareImageSchema
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FaceVerificationResponse(BaseModel):
    """Response for POST /api/v1/faces/verify.

    Mode B (external_id supplied): 1:1 verification against that identity's
    enrolled embeddings -- external_id is always echoed back since the caller
    supplied it themselves. Mode A (external_id omitted): 1:N identification
    against every enrolled identity -- external_id is only populated when the
    best match clears the threshold, and is null on FAIL so a low-confidence
    guess is never surfaced as an identity.
    """

    verified: bool
    status: Literal["PASS", "FAIL"]
    external_id: str | None
    similarity: float
    threshold: float
    mode: Literal["verification", "identification"]
    detection_score: float
    quality: QualityResultSchema
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FrameDiagnostic(BaseModel):
    """Per-frame outcome for a multi-frame verification request. Only
    populated in the response when the caller asks for debug detail."""

    frame_index: int
    valid: bool
    external_id: str | None = None
    similarity: float | None = None
    passed_threshold: bool | None = None
    detection_score: float | None = None
    quality: QualityResultSchema | None = None
    rejection_reason: str | None = None


class MultiFrameVerificationResponse(BaseModel):
    """Response for POST /api/v1/faces/verify-multi.

    The verdict combines two conditions, not one: the *same* identity must be
    the top match across enough of the valid frames (identity consistency),
    and enough of that identity's frames must *also* individually clear the
    similarity threshold (threshold agreement) -- `consensus_ratio` is the
    achieved fraction satisfying both at once, checked against
    `required_consensus_ratio` and a minimum absolute `frames_agreeing` floor
    so a single strong frame can never carry a PASS on its own.
    """

    verified: bool
    status: Literal["PASS", "FAIL"]
    external_id: str | None
    similarity: float | None
    threshold: float
    mode: Literal["verification", "identification"]
    frames_submitted: int
    frames_valid: int
    frames_agreeing: int
    consensus_ratio: float
    required_consensus_ratio: float
    reasons: list[str] = Field(default_factory=list)
    frames: list[FrameDiagnostic] | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
