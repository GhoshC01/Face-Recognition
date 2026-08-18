from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import FaceServiceError
from app.schemas.common import ErrorResponse
from app.utils.request_context import get_request_id

logger = logging.getLogger(__name__)


def _error_response(status_code: int, error_code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(
        error_code=error_code,
        message=message,
        request_id=get_request_id(),
        details=details or {},
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FaceServiceError)
    async def handle_face_service_error(_: Request, exc: FaceServiceError) -> JSONResponse:
        logger.warning(
            "face_service_error",
            extra={"error_code": exc.error_code, "error_message": exc.message},
        )
        return _error_response(exc.http_status, exc.error_code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            "Request validation failed",
            {"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception")
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_server_error",
            "An unexpected error occurred",
        )
