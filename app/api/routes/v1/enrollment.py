from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import get_enrollment_service, get_settings, require_api_key
from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.schemas.enrollment import EnrollmentDeleteResponse, EnrollmentResponse, EnrollmentStatusResponse
from app.services.enrollment_service import EnrollmentService
from app.utils.image_utils import decode_image_bytes

router = APIRouter(prefix="/enrollment", tags=["enrollment"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_face(
    external_id: str = Form(..., description="Opaque identity key supplied by the calling system"),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: EnrollmentService = Depends(get_enrollment_service),
) -> EnrollmentResponse:
    image_bytes = await read_validated_upload(file, settings)
    image = decode_image_bytes(image_bytes)
    return service.enroll(external_id, image)


@router.get("/{external_id}", response_model=EnrollmentStatusResponse)
def get_enrollment_status(
    external_id: str,
    service: EnrollmentService = Depends(get_enrollment_service),
) -> EnrollmentStatusResponse:
    return service.status(external_id)


@router.delete("/{external_id}", response_model=EnrollmentDeleteResponse)
def delete_enrollment(
    external_id: str,
    service: EnrollmentService = Depends(get_enrollment_service),
) -> EnrollmentDeleteResponse:
    return service.remove(external_id)
