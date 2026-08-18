from __future__ import annotations

import asyncio

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.schemas.common import ErrorResponse
from app.utils.request_context import get_request_id


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Bounds how long any single request may run, so a pathological image
    or a stalled ONNX call can't tie up a worker indefinitely."""

    def __init__(self, app, timeout_seconds: float) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            body = ErrorResponse(
                error_code="request_timeout",
                message=f"Request exceeded the {self.timeout_seconds}s timeout",
                request_id=get_request_id(),
            )
            return JSONResponse(status_code=504, content=body.model_dump(mode="json"))
