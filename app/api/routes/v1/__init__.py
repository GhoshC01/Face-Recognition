from fastapi import APIRouter

from app.api.routes.v1 import enrollment, faces, verification

api_router = APIRouter()
api_router.include_router(enrollment.router)
api_router.include_router(faces.router)
api_router.include_router(verification.router)
