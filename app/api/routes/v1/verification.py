from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import get_evaluation_service, get_settings, get_verification_service, require_api_key
from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.schemas.verification import CompareResponse, IdentificationResponse, VerificationResponse
from app.services.evaluation_service import EvaluationService
from app.services.verification_service import VerificationService
from app.utils.image_utils import decode_image_bytes

router = APIRouter(prefix="/verification", tags=["verification"], dependencies=[Depends(require_api_key)])


@router.post("/verify", response_model=VerificationResponse)
async def verify_face(
    external_id: str = Form(..., description="Identity to verify the captured face against"),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: VerificationService = Depends(get_verification_service),
) -> VerificationResponse:
    """1:1 verification: does the captured face match the claimed external_id?"""
    image_bytes = await read_validated_upload(file, settings)
    image = decode_image_bytes(image_bytes)
    return service.verify(external_id, image)


@router.post("/identify", response_model=IdentificationResponse)
async def identify_face(
    file: UploadFile = File(...),
    top_k: int | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    service: VerificationService = Depends(get_verification_service),
) -> IdentificationResponse:
    """1:N identification: search all enrolled identities for the best matches."""
    image_bytes = await read_validated_upload(file, settings)
    image = decode_image_bytes(image_bytes)
    return service.identify(image, top_k=top_k)


@router.post("/compare", response_model=CompareResponse, status_code=status.HTTP_200_OK)
async def compare_faces(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: EvaluationService = Depends(get_evaluation_service),
) -> CompareResponse:
    """Stateless 1:1 comparison of two images; no enrollment/storage involved."""
    image_a = decode_image_bytes(await read_validated_upload(file_a, settings))
    image_b = decode_image_bytes(await read_validated_upload(file_b, settings))
    return service.compare(image_a, image_b)
