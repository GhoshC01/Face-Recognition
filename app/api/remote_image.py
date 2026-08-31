from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Callable
from typing import Union
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from fastapi import UploadFile

from app.api.upload_validation import read_validated_upload
from app.config.settings import Settings
from app.core.exceptions import (
    InvalidImageSourceError,
    PayloadTooLargeError,
    RemoteImageFetchError,
    UnsupportedMediaTypeError,
)

logger = logging.getLogger(__name__)

# S3 objects are often stored without an image Content-Type. Treat these as
# "unknown binary" and let decode_image_bytes decide whether the bytes are
# a real image -- otherwise presigned S3 URLs get a false 415.
_OPAQUE_CONTENT_TYPES = frozenset(
    {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/x-www-form-urlencoded",
    }
)
_CONTENT_TYPE_ALIASES = {"image/jpg": "image/jpeg"}

# Type aliases are evaluated at import time, so `|` is not valid on Python 3.9.
IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
IpResolver = Callable[[str], list[IPAddress]]


def _redact_url(url: str) -> str:
    """Strip query/fragment so presigned S3 signatures never land in logs or errors."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _is_populated_upload(file: UploadFile | None) -> bool:
    return file is not None and bool(file.filename)


def _normalized_content_type(header: str | None) -> str:
    raw = (header or "").split(";", 1)[0].strip().lower()
    return _CONTENT_TYPE_ALIASES.get(raw, raw)


def _resolve_host_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RemoteImageFetchError(
            "Could not resolve image URL host",
            url=_redact_url(f"https://{hostname}/"),
        ) from exc

    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        ips.append(ipaddress.ip_address(info[4][0]))
    if not ips:
        raise RemoteImageFetchError("Image URL host resolved to no addresses", host=hostname)
    return ips


def _ips_for_host(
    hostname: str,
    resolve_ips: IpResolver | None = None,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        return [ipaddress.ip_address(hostname)]
    except ValueError:
        resolver = resolve_ips or _resolve_host_ips
        return resolver(hostname)


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def assert_public_http_url(url: str, resolve_ips: IpResolver | None = None) -> None:
    """Reject non-http(s) URLs, embedded credentials, and hosts that resolve
    to private/loopback/link-local addresses (SSRF). Public S3 / CloudFront
    hosts pass because they resolve to public IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RemoteImageFetchError("Image URL must be http or https", url=_redact_url(url))
    if not parsed.hostname:
        raise RemoteImageFetchError("Image URL is missing a host", url=_redact_url(url))
    if parsed.username or parsed.password:
        raise RemoteImageFetchError("Image URL must not contain credentials", url=_redact_url(url))

    for ip in _ips_for_host(parsed.hostname, resolve_ips=resolve_ips):
        if _is_blocked_ip(ip):
            raise RemoteImageFetchError(
                "Image URL host is not allowed",
                url=_redact_url(url),
            )


def _assert_remote_content_type(content_type: str | None, settings: Settings) -> None:
    normalized = _normalized_content_type(content_type)
    if normalized in _OPAQUE_CONTENT_TYPES:
        return
    if settings.allowed_content_types and normalized not in settings.allowed_content_types:
        raise UnsupportedMediaTypeError(
            f"Unsupported content type '{normalized}'",
            content_type=normalized,
            allowed_content_types=settings.allowed_content_types,
        )


async def fetch_remote_image(
    url: str,
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    resolve_ips: IpResolver | None = None,
) -> bytes:
    """GET a public image URL (S3 presigned, CloudFront, or any public HTTPS)
    and return raw bytes. Size and content-type limits match file uploads,
    except opaque S3 Content-Types are not rejected."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    timeout = httpx.Timeout(settings.remote_image_timeout_seconds)
    own_client = client is None
    http_client = client or httpx.AsyncClient(follow_redirects=False, timeout=timeout)

    current = url.strip()
    seen = 0
    try:
        while True:
            assert_public_http_url(current, resolve_ips=resolve_ips)
            try:
                response = await http_client.send(
                    http_client.build_request("GET", current),
                    stream=True,
                )
            except httpx.TimeoutException as exc:
                raise RemoteImageFetchError(
                    "Timed out fetching image URL",
                    url=_redact_url(current),
                ) from exc
            except httpx.HTTPError as exc:
                raise RemoteImageFetchError(
                    "Failed to fetch image URL",
                    url=_redact_url(current),
                ) from exc

            try:
                if response.has_redirect_location:
                    seen += 1
                    if seen > settings.remote_image_max_redirects:
                        raise RemoteImageFetchError(
                            "Too many redirects fetching image URL",
                            url=_redact_url(url),
                        )
                    current = urljoin(str(response.url), response.headers["location"])
                    continue

                if response.status_code >= 400:
                    raise RemoteImageFetchError(
                        f"Image URL returned HTTP {response.status_code}",
                        url=_redact_url(current),
                        status_code=response.status_code,
                    )

                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        if int(declared) > max_bytes:
                            raise PayloadTooLargeError(
                                f"Remote image exceeds the {settings.max_upload_size_mb}MB limit",
                                declared_size=int(declared),
                                max_bytes=max_bytes,
                            )
                    except ValueError:
                        pass

                _assert_remote_content_type(response.headers.get("content-type"), settings)

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise PayloadTooLargeError(
                            f"Remote image exceeds the {settings.max_upload_size_mb}MB limit",
                            actual_size=total,
                            max_bytes=max_bytes,
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                await response.aclose()
    finally:
        if own_client:
            await http_client.aclose()


async def resolve_image_bytes(
    file: UploadFile | None,
    url: str | None,
    settings: Settings,
    *,
    field_name: str,
    client: httpx.AsyncClient | None = None,
    resolve_ips: IpResolver | None = None,
) -> bytes:
    """Exactly one of a multipart file or a remote URL must be supplied."""
    has_file = _is_populated_upload(file)
    has_url = bool(url and url.strip())

    if has_file and has_url:
        raise InvalidImageSourceError(
            f"Provide either {field_name} as a file or {field_name}_url, not both",
            field=field_name,
        )
    if not has_file and not has_url:
        raise InvalidImageSourceError(
            f"Either {field_name} (file) or {field_name}_url is required",
            field=field_name,
        )

    if has_file:
        return await read_validated_upload(file, settings)

    logger.info("fetching_remote_compare_image", extra={"field": field_name, "url": _redact_url(url.strip())})
    return await fetch_remote_image(url.strip(), settings, client=client, resolve_ips=resolve_ips)
