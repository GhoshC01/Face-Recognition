from __future__ import annotations

import json

import cv2
import numpy as np


def _encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def _embedding(client) -> str:
    dimension = client.app.state.settings.embedding_dimension
    vector = [0.0] * dimension
    vector[0] = 1.0
    return json.dumps(vector)


def test_compare_embedding_invalid_image_returns_400(client):
    response = client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": _embedding(client)},
        files={"image": ("a.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"


def test_compare_embedding_without_models_returns_service_unavailable(client):
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    response = client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": _embedding(client)},
        files={"image": ("a.png", _encode_png(image), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_not_ready"


def test_compare_embedding_requires_image(client):
    response = client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": _embedding(client)},
    )

    assert response.status_code == 422
