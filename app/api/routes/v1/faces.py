from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_evaluation_service, get_settings, require_api_key
from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.core.exceptions import InvalidEmbeddingError
from app.schemas.verification import FaceCompareEmbeddingResponse
from app.services.evaluation_service import EvaluationService
from app.utils.image_utils import decode_image_bytes
from app.utils.timing import Stopwatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces", tags=["faces"], dependencies=[Depends(require_api_key)])


def _parse_embedding_form(raw: str) -> np.ndarray:
    """Parse a multipart form field that holds a JSON array of floats.

    Accepts plain JSON (`[-0.08, ...]`) or the SSMS-style hex of that same
    JSON text (`0x5B2D302E...`).
    """
    text = raw.strip()
    if text.lower().startswith("0x"):
        hex_body = text[2:]
        try:
            text = bytes.fromhex(hex_body).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidEmbeddingError(
                "embedding hex must be a 0x-prefixed encoding of a JSON array"
            ) from exc
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidEmbeddingError("embedding must be a JSON array of numbers") from exc
    if not isinstance(values, list) or not values:
        raise InvalidEmbeddingError("embedding must be a non-empty JSON array of numbers")
    try:
        return np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingError("embedding must be a JSON array of numbers") from exc


@router.post("/compare-embedding", response_model=FaceCompareEmbeddingResponse)
async def compare_face_with_embedding(
    embedding: str = Form(
        ...,
        description=(
            "Face embedding as a JSON float array, or the same JSON as SSMS-style "
            "hex (0x5B2D...). Length must match EMBEDDING_DIMENSION."
        ),
    ),
    image: UploadFile = File(..., description="Face image to compare against the supplied embedding."),
    settings: Settings = Depends(get_settings),
    service: EvaluationService = Depends(get_evaluation_service),
) -> FaceCompareEmbeddingResponse:
    """Stateless 1:1 compare: supplied embedding vs one image.

    Embeds the image (exactly one face, same live-capture quality gate as
    compare's image2), then cosine-compares against the given vector.
    Returns Match / Not matching + confidence.
    Nothing is enrolled or written to FAISS.
    """
    sw = Stopwatch()
    reference = _parse_embedding_form(embedding)
    decoded = decode_image_bytes(await read_validated_upload(image, settings))
    receive_ms = sw.lap_ms()
    try:
        return service.compare_with_embedding(
            reference,
            decoded,
            expected_dimension=settings.embedding_dimension,
        )
    finally:
        logger.info(
            "compare_embedding_route_timings",
            extra={"receive_ms": receive_ms, "route_total_ms": sw.lap_ms() + receive_ms},
        )
