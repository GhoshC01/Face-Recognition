from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.alignment import _ARCFACE_TEMPLATE
from app.core.detector import Face


def _default_face() -> Face:
    """A single, valid, non-degenerate detection: landmarks sit comfortably
    inside the box, which sits comfortably inside a 300x300 frame."""
    return Face(
        box=(20, 20, 280, 280),
        score=0.97,
        landmarks=_ARCFACE_TEMPLATE + np.array([20.0, 30.0], dtype=np.float32),
    )


class _FakeDetector:
    """Deterministic stand-in for SCRFD: returns a configurable, fixed list
    of detections regardless of image content, so the rest of the real
    pipeline (quality checks, alignment, FAISS) runs end-to-end over genuine
    HTTP requests without an actual ONNX model."""

    def __init__(self) -> None:
        self.is_loaded = True
        self.faces_to_return: list[Face] = [_default_face()]
        self._queue: list[list[Face]] = []

    def queue(self, *face_lists: list[Face]) -> None:
        """Script a different detection result per call, in order -- needed
        to simulate one bad frame among several good ones within a single
        multi-frame request. Falls back to `faces_to_return` once exhausted."""
        self._queue.extend(face_lists)

    def detect(self, image: np.ndarray) -> list[Face]:
        if self._queue:
            return self._queue.pop(0)
        return self.faces_to_return


class _FakeEmbedder:
    """Deterministic stand-in for MobileFaceNet: returns pre-queued embedding
    vectors in call order, decoupled from pixel content. This lets each test
    control exactly what "identity" a request encodes while every other
    stage -- quality gating, real alignment warp, real FAISS storage/search,
    exception handling, response schemas -- still runs for real."""

    def __init__(self, dimension: int = 512) -> None:
        self.is_loaded = True
        self.input_size = (112, 112)
        self.dimension = dimension
        self._queue: list[np.ndarray] = []

    def queue(self, *embeddings: np.ndarray) -> None:
        self._queue.extend(embeddings)

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        assert self._queue, "FakeEmbedder.embed() called with no queued embedding left"
        return self._queue.pop(0)


def _unit_vector(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dimension).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _nudged(base: np.ndarray, seed: int, noise_scale: float = 0.05) -> np.ndarray:
    """A vector close to `base` -- simulates a second, imperfect capture of
    the same enrolled identity."""
    rng = np.random.default_rng(seed)
    nudged = base + rng.normal(scale=noise_scale, size=base.shape).astype(np.float32)
    return nudged / np.linalg.norm(nudged)


def _valid_face_image(size: int = 300) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(90, 170, size=(size, size, 3), dtype=np.uint8)


def _encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def rigged_client(client):
    """`client` boots the full real app; models fail to load (no ONNX
    binaries in the test environment) so app.state.detector/embedder start
    out real-but-unloaded. Swap in fakes directly on the already-constructed
    recognizer -- the enrollment/verification services hold a reference to
    that same object -- so routing, dependency injection, quality checks,
    alignment, and FAISS all still execute for real."""
    fake_detector = _FakeDetector()
    fake_embedder = _FakeEmbedder(dimension=512)
    client.app.state.recognizer.detector = fake_detector
    client.app.state.recognizer.embedder = fake_embedder
    client.app.state.detector = fake_detector
    client.app.state.embedder = fake_embedder
    return client


def test_compare_embedding_same_person_returns_match(rigged_client):
    import json

    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(_nudged(base, seed=2))

    response = rigged_client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": json.dumps(base.tolist())},
        files={"image": ("probe.png", _encode_png(_valid_face_image()), "image/png")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "Match"
    assert body["matched"] is True
    assert body["confidence"] >= body["threshold"]
    assert "image" in body
    assert "detection_score" in body["image"]


def test_compare_embedding_different_person_returns_not_matching(rigged_client):
    import json

    embedder = rigged_client.app.state.recognizer.embedder
    reference = _unit_vector(512, seed=1)
    embedder.queue(_unit_vector(512, seed=99))

    response = rigged_client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": json.dumps(reference.tolist())},
        files={"image": ("probe.png", _encode_png(_valid_face_image()), "image/png")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "Not matching"
    assert body["matched"] is False
    assert body["confidence"] < body["threshold"]


def test_compare_embedding_rejects_wrong_dimension(rigged_client):
    import json

    response = rigged_client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": json.dumps([0.1, 0.2, 0.3])},
        files={"image": ("probe.png", _encode_png(_valid_face_image()), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_embedding"


def test_compare_embedding_rejects_invalid_json(rigged_client):
    response = rigged_client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": "not-json"},
        files={"image": ("probe.png", _encode_png(_valid_face_image()), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_embedding"


def test_compare_embedding_accepts_ssms_hex(rigged_client):
    import json

    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(_nudged(base, seed=2))
    hex_payload = "0x" + json.dumps(base.tolist()).encode("utf-8").hex()

    response = rigged_client.post(
        "/api/v1/faces/compare-embedding",
        data={"embedding": hex_payload},
        files={"image": ("probe.png", _encode_png(_valid_face_image()), "image/png")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "Match"

