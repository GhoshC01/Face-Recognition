from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import get_enrollment_service, get_settings, get_verification_service, require_api_key
from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.core.exceptions import FaceServiceError
from app.schemas.enrollment import EnrollmentResponse
from app.schemas.verification import FaceVerificationResponse, MultiFrameVerificationResponse
from app.services.enrollment_service import EnrollmentService
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
