from __future__ import annotations

from fastapi import UploadFile

from app.config.settings import Settings
from app.core.exceptions import PayloadTooLargeError, UnsupportedMediaTypeError


async def read_validated_upload(file: UploadFile, settings: Settings) -> bytes:
    """Validates an upload's declared content type and size, then returns
    its raw bytes. This is an HTTP-boundary check -- it knows about upload
    metadata and configured limits, nothing about faces or embeddings -- so
    it runs before any image decoding or recognition is attempted.

    Malformed/corrupt file *content* that nonetheless has an allowed
    extension and size is still caught downstream by `decode_image_bytes`.
    """
    if settings.allowed_content_types and file.content_type not in settings.allowed_content_types:
        raise UnsupportedMediaTypeError(
            f"Unsupported content type '{file.content_type}'",
            content_type=file.content_type,
            allowed_content_types=settings.allowed_content_types,
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # `UploadFile.size`, when available, reflects what was already buffered
    # during multipart parsing -- checking it first fails fast without an
    # extra read, but a client can misreport it, so the actual byte count is
    # always verified below regardless.
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > max_bytes:
        raise PayloadTooLargeError(
            f"Upload exceeds the {settings.max_upload_size_mb}MB limit",
            declared_size=declared_size,
            max_bytes=max_bytes,
        )

    data = await file.read()
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"Upload exceeds the {settings.max_upload_size_mb}MB limit",
            actual_size=len(data),
            max_bytes=max_bytes,
        )

    return data
