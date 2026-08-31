from __future__ import annotations

import asyncio
import io
import ipaddress

import httpx
import pytest
from fastapi import UploadFile

from app.api.remote_image import assert_public_http_url, fetch_remote_image, resolve_image_bytes
from app.config.settings import Settings
from app.core.exceptions import (
    InvalidImageSourceError,
    PayloadTooLargeError,
    RemoteImageFetchError,
    UnsupportedMediaTypeError,
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
    b"\xc0\x00\x00\x00\x03\x00\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

S3_URL = (
    "https://my-bucket.s3.ap-south-1.amazonaws.com/faces/a.jpg"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc"
)


def _settings(**overrides) -> Settings:
    defaults = dict(
        max_upload_size_mb=1,
        allowed_content_types=["image/jpeg", "image/png", "image/webp"],
        remote_image_timeout_seconds=5.0,
        remote_image_max_redirects=3,
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _public_ip(_host: str):
    return [ipaddress.ip_address("3.108.0.1")]


def test_rejects_loopback_url():
    with pytest.raises(RemoteImageFetchError):
        assert_public_http_url("http://127.0.0.1/secret.png")


def test_rejects_link_local_metadata_url():
    with pytest.raises(RemoteImageFetchError):
        assert_public_http_url("http://169.254.169.254/latest/meta-data")


def test_rejects_non_http_scheme():
    with pytest.raises(RemoteImageFetchError):
        assert_public_http_url("file:///etc/passwd")


def test_rejects_embedded_credentials():
    with pytest.raises(RemoteImageFetchError):
        assert_public_http_url("https://user:pass@example.com/a.png")


def test_allows_s3_style_host_with_public_ip():
    assert_public_http_url(S3_URL, resolve_ips=_public_ip)


def _run(coro):
    return asyncio.run(coro)


def test_fetch_presigned_s3_url_keeps_query_and_accepts_octet_stream():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            content=PNG_BYTES,
            headers={"content-type": "application/octet-stream"},
        )

    async def _go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            return await fetch_remote_image(
                S3_URL, _settings(), client=client, resolve_ips=_public_ip
            )

    data = _run(_go())
    assert data == PNG_BYTES
    assert "X-Amz-Signature=abc" in seen["url"]


def test_fetch_accepts_image_jpg_alias():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PNG_BYTES, headers={"content-type": "image/jpg"})

    async def _go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            return await fetch_remote_image(
                "https://cdn.example.com/a.jpg", _settings(), client=client, resolve_ips=_public_ip
            )

    assert _run(_go()) == PNG_BYTES


def test_fetch_rejects_html_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html></html>", headers={"content-type": "text/html"})

    async def _go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            await fetch_remote_image(
                "https://cdn.example.com/a.jpg", _settings(), client=client, resolve_ips=_public_ip
            )

    with pytest.raises(UnsupportedMediaTypeError):
        _run(_go())


def test_fetch_rejects_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=b"expired")

    async def _go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            await fetch_remote_image(
                S3_URL, _settings(), client=client, resolve_ips=_public_ip
            )

    with pytest.raises(RemoteImageFetchError, match="HTTP 403"):
        _run(_go())


def test_fetch_rejects_oversized_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * (2 * 1024 * 1024),
            headers={"content-type": "image/png"},
        )

    async def _go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            await fetch_remote_image(
                "https://cdn.example.com/big.png",
                _settings(max_upload_size_mb=1),
                client=client,
                resolve_ips=_public_ip,
            )

    with pytest.raises(PayloadTooLargeError):
        _run(_go())


def test_fetch_follows_redirect_to_public_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/go":
            return httpx.Response(302, headers={"location": "/final.png"})
        return httpx.Response(200, content=PNG_BYTES, headers={"content-type": "image/png"})

    async def _go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            return await fetch_remote_image(
                "https://cdn.example.com/go", _settings(), client=client, resolve_ips=_public_ip
            )

    assert _run(_go()) == PNG_BYTES


def test_resolve_requires_file_or_url():
    async def _go():
        await resolve_image_bytes(None, None, _settings(), field_name="image1")

    with pytest.raises(InvalidImageSourceError, match="image1"):
        _run(_go())


def test_resolve_rejects_file_and_url_together():
    upload = UploadFile(filename="a.png", file=io.BytesIO(PNG_BYTES), headers={"content-type": "image/png"})

    async def _go():
        await resolve_image_bytes(upload, S3_URL, _settings(), field_name="image1")

    with pytest.raises(InvalidImageSourceError, match="not both"):
        _run(_go())
