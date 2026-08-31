from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import (
    get_enrollment_service,
    get_evaluation_service,
    get_settings,
    get_verification_service,
    require_api_key,
)
from app.api.remote_image import resolve_image_bytes
from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.core.exceptions import FaceServiceError
from app.schemas.enrollment import EnrollmentResponse
from app.schemas.verification import FaceCompareResponse, FaceVerificationResponse, MultiFrameVerificationResponse
from app.services.enrollment_service import EnrollmentService
from app.services.evaluation_service import EvaluationService
from app.services.verification_service import VerificationService
from app.utils.image_utils import decode_image_bytes
from app.utils.timing import Stopwatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces", tags=["faces"], dependencies=[Depends(require_api_key)])


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
    first = decode_image_bytes(raw1)
    second = decode_image_bytes(raw2)
    receive_ms = sw.lap_ms()
    try:
        return service.compare_pair(first, second)
    finally:
        logger.info(
            "compare_route_timings",
            extra={"receive_ms": receive_ms, "route_total_ms": sw.lap_ms() + receive_ms},
        )
