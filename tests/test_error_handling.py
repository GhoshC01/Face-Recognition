from __future__ import annotations

import cv2
import numpy as np


def _encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_compare_without_models_returns_service_unavailable(client):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    png_bytes = _encode_png(image)

    response = client.post(
        "/api/v1/verification/compare",
        files={
            "file_a": ("a.png", png_bytes, "image/png"),
            "file_b": ("b.png", png_bytes, "image/png"),
        },
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "model_not_ready"


def test_verify_unknown_identity_returns_404(client):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    png_bytes = _encode_png(image)

    response = client.post(
        "/api/v1/verification/verify",
        data={"external_id": "ghost"},
        files={"file": ("a.png", png_bytes, "image/png")},
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "identity_not_found"


def test_enroll_invalid_image_returns_400(client):
    response = client.post(
        "/api/v1/enrollment",
        data={"external_id": "emp-1"},
        files={"file": ("a.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"


def test_faces_enroll_without_models_returns_service_unavailable(client):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    png_bytes = _encode_png(image)

    response = client.post(
        "/api/v1/faces/enroll",
        data={"external_id": "EMP001"},
        files={"image": ("a.png", png_bytes, "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_not_ready"


def test_faces_enroll_requires_image(client):
    response = client.post(
        "/api/v1/faces/enroll",
        data={"external_id": "EMP001"},
    )

    assert response.status_code == 422  # FastAPI request validation: image is required


def test_faces_verify_multi_without_models_returns_service_unavailable(client):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    png_bytes = _encode_png(image)

    # Mode A (no external_id): unlike Mode B, there's no identity lookup to
    # fail fast on, so this actually reaches the recognizer and surfaces the
    # missing-model condition.
    response = client.post(
        "/api/v1/faces/verify-multi",
        files=[
            ("files", ("a.png", png_bytes, "image/png")),
            ("files", ("b.png", png_bytes, "image/png")),
            ("files", ("c.png", png_bytes, "image/png")),
        ],
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_not_ready"


def test_faces_verify_multi_rejects_too_few_frames(client):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    png_bytes = _encode_png(image)

    response = client.post(
        "/api/v1/faces/verify-multi",
        data={"external_id": "EMP001"},
        files=[("files", ("a.png", png_bytes, "image/png"))],
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_frame_count"
