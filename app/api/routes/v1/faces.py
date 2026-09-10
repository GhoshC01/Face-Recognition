from __future__ import annotations

import asyncio
import json
import logging

import httpx
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import (
    get_embedding_service,
    get_enrollment_service,
    get_evaluation_service,
    get_settings,
    get_verification_service,
    require_api_key,
)
from app.api.remote_image import resolve_image_bytes, resolve_image_list
from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.core.exceptions import FaceServiceError, InvalidEmbeddingError, InvalidImageError
from app.schemas.embedding import EmbedResponse
from app.schemas.enrollment import EnrollmentResponse
from app.schemas.verification import (
    FaceCompareEmbeddingResponse,
    FaceCompareResponse,
    FaceVerificationResponse,
    MultiFrameVerificationResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.enrollment_service import EnrollmentService
from app.services.evaluation_service import EvaluationService, raise_if_compare_sides_failed
from app.services.verification_service import VerificationService
from app.utils.image_utils import decode_image_bytes
from app.utils.timing import Stopwatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces", tags=["faces"], dependencies=[Depends(require_api_key)])


def _try_decode(raw: bytes):
    try:
        return decode_image_bytes(raw), None
    except InvalidImageError as exc:
        return None, exc


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


@router.post("/enroll", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_faces(
    external_id: str = Form(..., description="Opaque identity key supplied by the calling system, e.g. an HRMS employee ID"),
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: EnrollmentService = Depends(get_enrollment_service),
) -> EnrollmentResponse:
    """Initial enrollment workflow: a single image of the person. The image
    must contain exactly one detectable face and pass quality checks."""
    sw = Stopwatch()
    img = decode_image_bytes(await read_validated_upload(image, settings))
    receive_ms = sw.lap_ms()
    try:
        return service.enroll_initial(external_id, img)
    finally:
        logger.info(
            "enroll_route_timings",
            extra={"external_id": external_id, "receive_ms": receive_ms, "route_total_ms": sw.lap_ms() + receive_ms},
        )


@router.post("/verify", response_model=FaceVerificationResponse)
async def verify_face(
    file: UploadFile = File(...),
    external_id: str | None = Form(
        default=None,
        description=(
            "If provided, verify the image against this specific identity's enrolled "
            "embeddings (Mode B). If omitted, identify the best matching identity across "
            "everyone enrolled (Mode A)."
        ),
    ),
    settings: Settings = Depends(get_settings),
    service: VerificationService = Depends(get_verification_service),
) -> FaceVerificationResponse:
    """The main face verification endpoint HRMS calls. Requires exactly one
    detectable face in the image; never marks attendance or touches HRMS data
    -- it only returns PASS/FAIL plus the similarity/threshold behind it."""
    sw = Stopwatch()
    image = decode_image_bytes(await read_validated_upload(file, settings))
    receive_ms = sw.lap_ms()
    try:
        return service.verify_or_identify(image, external_id=external_id)
    finally:
        logger.info(
            "verify_route_timings",
            extra={"external_id": external_id, "receive_ms": receive_ms, "route_total_ms": sw.lap_ms() + receive_ms},
        )


@router.post("/verify-multi", response_model=MultiFrameVerificationResponse)
async def verify_faces_multi_frame(
    files: list[UploadFile] = File(..., description="Several frames captured in quick succession, e.g. 3-5 webcam captures"),
    external_id: str | None = Form(
        default=None,
        description="If provided, verify against this specific identity (Mode B). If omitted, identify the best matching identity (Mode A).",
    ),
    debug: bool = Form(default=False, description="If true, include per-frame diagnostics in the response"),
    settings: Settings = Depends(get_settings),
    service: VerificationService = Depends(get_verification_service),
) -> MultiFrameVerificationResponse:
    """Multi-frame verification for webcam-based attendance capture: reduces
    false recognition caused by any single blurry, occluded, or otherwise
    poor-quality frame. A frame that fails to decode, has no/multiple faces,
    or fails quality checks is ignored rather than failing the whole
    request; the final PASS/FAIL requires enough of the remaining valid
    frames to agree on the same identity and individually clear the
    similarity threshold. Never marks attendance -- HRMS owns everything
    downstream of the returned verdict."""
    images = []
    for upload in files:
        try:
            raw = await read_validated_upload(upload, settings)
            images.append(decode_image_bytes(raw))
        except FaceServiceError:
            # An unsupported type, oversized, or corrupt frame is ignored
            # like any other bad frame -- it never fails the whole request.
            images.append(None)

    return service.verify_multi_frame(images, external_id=external_id, debug=debug)


@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(
    image1: UploadFile | None = File(
        default=None,
        description="First face image as a file upload. Supply this or image1_url, not both.",
    ),
    image2: UploadFile | None = File(
        default=None,
        description="Second face image as a file upload. Supply this or image2_url, not both.",
    ),
    image1_url: str | None = Form(
        default=None,
        description="Public or S3 presigned HTTPS URL for the first image. The service downloads it.",
    ),
    image2_url: str | None = Form(
        default=None,
        description="Public or S3 presigned HTTPS URL for the second image. The service downloads it.",
    ),
    settings: Settings = Depends(get_settings),
    service: EvaluationService = Depends(get_evaluation_service),
) -> FaceCompareResponse:
    """Stateless 1:1 comparison of two images. Each side is either a
    multipart file (`image1` / `image2`) or a remote URL (`image1_url` /
    `image2_url`, including S3 presigned links). The service fetches URLs
    itself so the caller does not need to download from S3 first. Each
    image must contain exactly one detectable face."""
    sw = Stopwatch()
    timeout = httpx.Timeout(settings.remote_image_timeout_seconds)
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        raw1, raw2 = await asyncio.gather(
            resolve_image_bytes(image1, image1_url, settings, field_name="image1", client=client),
            resolve_image_bytes(image2, image2_url, settings, field_name="image2", client=client),
        )
    first, decode_error1 = _try_decode(raw1)
    second, decode_error2 = _try_decode(raw2)
    raise_if_compare_sides_failed(decode_error1, decode_error2)
    assert first is not None and second is not None
    receive_ms = sw.lap_ms()
    try:
        return service.compare_pair(first, second)
    finally:
        logger.info(
            "compare_route_timings",
            extra={"receive_ms": receive_ms, "route_total_ms": sw.lap_ms() + receive_ms},
        )


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
    Returns Match / Not matching + confidence like `/faces/compare`.
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


@router.post("/embed", response_model=EmbedResponse)
async def embed_faces(
    images: list[UploadFile] | None = File(
        default=None,
        description="One or more face images as file uploads. Supply this or image_urls, not both.",
    ),
    image_urls: list[str] | None = Form(
        default=None,
        description="One or more public or S3 presigned HTTPS URLs. Supply this or images, not both.",
    ),
    settings: Settings = Depends(get_settings),
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedResponse:
    """Stateless embedding extraction. Accepts one image or several.

    Each image must contain exactly one detectable face and pass quality
    checks. When more than one image is sent, every later image is compared
    against the first; if any pair falls below EMBED_CONSISTENCY_THRESHOLD
    the request fails with `inconsistent_images` and no vectors are
    returned. Nothing is enrolled or written to FAISS.
    """
    sw = Stopwatch()
    raw_images = await resolve_image_list(
        images,
        image_urls,
        settings,
        max_images=settings.embed_max_images,
    )
    decoded = []
    for index, raw in enumerate(raw_images):
        try:
            decoded.append(decode_image_bytes(raw))
        except InvalidImageError as exc:
            raise InvalidImageError(
                f"image[{index}]: {exc.message}",
                image_index=index,
                **exc.details,
            ) from exc
    receive_ms = sw.lap_ms()
    try:
        return service.embed(decoded)
    finally:
        logger.info(
            "embed_route_timings",
            extra={
                "images": len(decoded),
                "receive_ms": receive_ms,
                "route_total_ms": sw.lap_ms() + receive_ms,
            },
        )
