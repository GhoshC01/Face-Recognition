from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.core.exceptions import PayloadTooLargeError, UnsupportedMediaTypeError


def _upload(content: bytes, content_type: str = "image/png") -> UploadFile:
    return UploadFile(filename="probe.png", file=io.BytesIO(content), headers={"content-type": content_type})


def _settings(**overrides) -> Settings:
    defaults = dict(max_upload_size_mb=1, allowed_content_types=["image/jpeg", "image/png", "image/webp"])
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.mark.anyio
async def test_accepts_a_valid_upload_within_limits():
    data = b"x" * 100
    result = await read_validated_upload(_upload(data), _settings())
    assert result == data


@pytest.mark.anyio
async def test_rejects_unsupported_content_type():
    with pytest.raises(UnsupportedMediaTypeError):
        await read_validated_upload(_upload(b"data", content_type="text/plain"), _settings())


@pytest.mark.anyio
async def test_rejects_oversized_upload():
    settings = _settings(max_upload_size_mb=1)
    oversized = b"x" * (2 * 1024 * 1024)  # 2MB > 1MB limit

    with pytest.raises(PayloadTooLargeError):
        await read_validated_upload(_upload(oversized), settings)


@pytest.mark.anyio
async def test_allows_any_content_type_when_allowlist_is_empty():
    settings = _settings(allowed_content_types=[])
    result = await read_validated_upload(_upload(b"data", content_type="text/plain"), settings)
    assert result == b"data"
