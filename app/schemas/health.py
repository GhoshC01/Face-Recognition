from __future__ import annotations

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str
    detector_loaded: bool
    recognizer_loaded: bool
    vector_store_ready: bool
    enrolled_identities: int
    app_version: str
