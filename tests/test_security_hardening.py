from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import create_app


def _encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _build_client(monkeypatch, tmp_path, **extra_env: str) -> TestClient:
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("METADATA_DIR", str(tmp_path / "metadata"))
    monkeypatch.setenv("DETECTOR_MODEL_PATH", str(tmp_path / "missing_det.onnx"))
    monkeypatch.setenv("RECOGNIZER_MODEL_PATH", str(tmp_path / "missing_rec.onnx"))
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return TestClient(create_app())


# --- upload validation enforced over real HTTP ---


def test_rejects_unsupported_content_type_over_http(client):
    response = client.post(
        "/api/v1/enrollment",
        data={"external_id": "emp-1"},
        files={"file": ("probe.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error_code"] == "unsupported_media_type"


def test_rejects_oversized_upload_over_http(client):
    oversized = b"\x00" * (9 * 1024 * 1024)  # default limit is 8MB

    response = client.post(
        "/api/v1/enrollment",
        data={"external_id": "emp-1"},
        files={"file": ("probe.png", oversized, "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


def test_valid_small_image_still_reaches_normal_processing(client):
    # No models loaded in this fixture -> should get past upload validation
    # and fail downstream with model_not_ready, not a validation error.
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    response = client.post(
        "/api/v1/enrollment",
        data={"external_id": "emp-1"},
        files={"file": ("probe.png", _encode_png(image), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_not_ready"


# --- CORS defaults to off; never combines wildcard with credentials ---


def test_cors_headers_absent_by_default(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path)
    with client:
        response = client.get("/health/live", headers={"Origin": "https://example.com"})

    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_cors_headers_present_when_explicitly_configured(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, CORS_ALLOW_ORIGINS=json.dumps(["https://hrms.example.com"]))
    with client:
        response = client.get("/health/live", headers={"Origin": "https://hrms.example.com"})

    assert response.headers.get("access-control-allow-origin") == "https://hrms.example.com"


# --- rate limiting ---


def test_rate_limit_blocks_after_configured_threshold(monkeypatch, tmp_path):
    client = _build_client(
        monkeypatch,
        tmp_path,
        RATE_LIMIT_ENABLED="true",
        RATE_LIMIT_REQUESTS="2",
        RATE_LIMIT_WINDOW_SECONDS="60",
    )
    with client:
        first = client.get("/health/live")
        second = client.get("/health/live")
        third = client.get("/health/live")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error_code"] == "rate_limit_exceeded"


def test_rate_limit_does_not_apply_when_disabled(monkeypatch, tmp_path):
    client = _build_client(monkeypatch, tmp_path, RATE_LIMIT_ENABLED="false")
    with client:
        responses = [client.get("/health/live") for _ in range(10)]

    assert all(r.status_code == 200 for r in responses)


# --- model file paths never leak into API responses ---


def test_model_path_never_appears_in_error_response(client, tmp_path):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    response = client.post(
        "/api/v1/enrollment",
        data={"external_id": "emp-1"},
        files={"file": ("probe.png", _encode_png(image), "image/png")},
    )

    assert response.status_code == 503
    body_text = response.text
    assert str(tmp_path) not in body_text
    assert "missing_det.onnx" not in body_text
    assert "missing_rec.onnx" not in body_text


# --- API key placeholder guard refuses to start in production ---


def test_refuses_to_start_in_production_with_default_api_key(monkeypatch, tmp_path):
    client = _build_client(
        monkeypatch,
        tmp_path,
        ENVIRONMENT="production",
        API_KEY_ENABLED="true",
        API_KEY="changeme",
    )

    with pytest.raises(RuntimeError):
        with client:
            pass


def test_allows_default_api_key_outside_production_with_warning(monkeypatch, tmp_path):
    client = _build_client(
        monkeypatch,
        tmp_path,
        ENVIRONMENT="development",
        API_KEY_ENABLED="true",
        API_KEY="changeme",
    )

    with client:
        response = client.get("/health/live")

    assert response.status_code == 200
