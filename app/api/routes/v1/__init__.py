from fastapi import APIRouter

from app.api.routes.v1 import faces

api_router = APIRouter()
api_router.include_router(faces.router)
