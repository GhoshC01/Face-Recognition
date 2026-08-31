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


def _enroll(client, external_id: str):
    image_bytes = _encode_png(_valid_face_image())
    return client.post(
        "/api/v1/faces/enroll",
        data={"external_id": external_id},
        files={"image": ("a.png", image_bytes, "image/png")},
    )


def _verify(client, external_id: str | None = None):
    image_bytes = _encode_png(_valid_face_image())
    data = {"external_id": external_id} if external_id is not None else {}
    return client.post(
        "/api/v1/faces/verify",
        data=data,
        files={"file": ("probe.png", image_bytes, "image/png")},
    )


# --- Mode B: 1:1 verification against a claimed external_id ---


def test_enroll_then_verify_mode_b_pass(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base)

    enroll_response = _enroll(rigged_client, "EMP001")
    assert enroll_response.status_code == 201, enroll_response.text

    embedder.queue(_nudged(base, seed=3))
    response = _verify(rigged_client, external_id="EMP001")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PASS"
    assert body["verified"] is True
    assert body["external_id"] == "EMP001"
    assert body["mode"] == "verification"
    assert body["similarity"] >= body["threshold"]


def test_verify_mode_b_fail_for_different_person(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base)
    _enroll(rigged_client, "EMP001")

    stranger = _unit_vector(512, seed=42)  # unrelated direction -> low similarity
    embedder.queue(stranger)
    response = _verify(rigged_client, external_id="EMP001")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAIL"
    assert body["verified"] is False
    # the claimed identity is echoed back even on FAIL -- the caller supplied it themselves
    assert body["external_id"] == "EMP001"
    assert body["similarity"] < body["threshold"]


def test_verify_unknown_external_id_returns_404(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    embedder.queue(_unit_vector(512, seed=1))

    response = _verify(rigged_client, external_id="ghost")

    assert response.status_code == 404
    assert response.json()["error_code"] == "identity_not_found"


# --- Mode A: 1:N identification, no external_id supplied ---


def test_verify_mode_a_identification_pass(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base)
    _enroll(rigged_client, "EMP001")

    embedder.queue(_nudged(base, seed=3))
    response = _verify(rigged_client)  # no external_id -> Mode A

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PASS"
    assert body["mode"] == "identification"
    assert body["external_id"] == "EMP001"


def test_verify_mode_a_fail_returns_null_external_id(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base)
    _enroll(rigged_client, "EMP001")

    embedder.queue(_unit_vector(512, seed=42))  # unrelated -> below threshold
    response = _verify(rigged_client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAIL"
    assert body["verified"] is False
    assert body["external_id"] is None  # never surface a low-confidence guess as an identity


def test_verify_mode_a_with_nothing_enrolled_returns_fail(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    embedder.queue(_unit_vector(512, seed=1))

    response = _verify(rigged_client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAIL"
    assert body["external_id"] is None


# --- exactly one face is required ---


def test_verify_rejects_multiple_faces(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base)
    _enroll(rigged_client, "EMP001")

    rigged_client.app.state.recognizer.detector.faces_to_return = [_default_face(), _default_face()]

    response = _verify(rigged_client, external_id="EMP001")

    assert response.status_code == 422
    assert response.json()["error_code"] == "multiple_faces_detected"


def test_verify_rejects_no_face_image(rigged_client):
    rigged_client.app.state.recognizer.detector.faces_to_return = []

    response = _verify(rigged_client, external_id="EMP001")

    assert response.status_code == 422
    assert response.json()["error_code"] == "no_face_detected"


# --- multi-frame verification ---


def _verify_multi(client, num_frames: int, external_id: str | None = None, debug: bool = False):
    files = [("files", (f"probe{i}.png", _encode_png(_valid_face_image()), "image/png")) for i in range(num_frames)]
    data: dict[str, str] = {}
    if external_id is not None:
        data["external_id"] = external_id
    if debug:
        data["debug"] = "true"
    return client.post("/api/v1/faces/verify-multi", data=data, files=files)


def test_verify_multi_pass_with_consistent_frames(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base)
    _enroll(rigged_client, "EMP001")

    embedder.queue(_nudged(base, seed=3), _nudged(base, seed=4), _nudged(base, seed=5))
    response = _verify_multi(rigged_client, num_frames=3, external_id="EMP001")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PASS"
    assert body["external_id"] == "EMP001"
    assert body["frames_submitted"] == 3
    assert body["frames_valid"] == 3
    assert body["frames_agreeing"] == 3
    assert body["frames"] is None  # debug not requested


def test_verify_multi_ignores_a_no_face_frame_and_still_passes(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    detector = rigged_client.app.state.recognizer.detector
    base = _unit_vector(512, seed=1)
    embedder.queue(base)
    _enroll(rigged_client, "EMP001")

    # 3 frames requested; the middle one detects no face and is skipped, so
    # embed() is only invoked for frames 0 and 2.
    detector.queue([_default_face()], [], [_default_face()])
    embedder.queue(_nudged(base, seed=3), _nudged(base, seed=4))

    response = _verify_multi(rigged_client, num_frames=3, external_id="EMP001", debug=True)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PASS"
    assert body["frames_submitted"] == 3
    assert body["frames_valid"] == 2
    assert body["frames_agreeing"] == 2
    assert body["frames"][1]["valid"] is False
    assert body["frames"][1]["rejection_reason"] == "no_face_detected"


# --- two-image compare (no enrollment) ---


def _compare(client):
    image_bytes = _encode_png(_valid_face_image())
    return client.post(
        "/api/v1/faces/compare",
        files={
            "image1": ("a.png", image_bytes, "image/png"),
            "image2": ("b.png", image_bytes, "image/png"),
        },
    )


def test_compare_same_person_returns_match(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base, _nudged(base, seed=2))

    response = _compare(rigged_client)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "Match"
    assert body["matched"] is True
    assert body["confidence"] >= body["threshold"]
    assert "image1" in body and "image2" in body


def test_compare_different_people_returns_not_matching(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    embedder.queue(_unit_vector(512, seed=1), _unit_vector(512, seed=99))

    response = _compare(rigged_client)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "Not matching"
    assert body["matched"] is False
    assert body["confidence"] < body["threshold"]


def test_compare_rejects_multiple_faces(rigged_client):
    embedder = rigged_client.app.state.recognizer.embedder
    embedder.queue(_unit_vector(512, seed=1))
    rigged_client.app.state.recognizer.detector.faces_to_return = [_default_face(), _default_face()]

    response = _compare(rigged_client)

    assert response.status_code == 422
    assert response.json()["error_code"] == "multiple_faces_detected"


def test_compare_rejects_no_face_image(rigged_client):
    rigged_client.app.state.recognizer.detector.faces_to_return = []

    response = _compare(rigged_client)

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "no_face_detected"
    assert body["details"]["failed_images"] == ["image1", "image2"]
    assert body["details"]["image1"]["ok"] is False
    assert body["details"]["image2"]["ok"] is False
    assert "image1:" in body["message"]
    assert "image2:" in body["message"]


def _too_small_face() -> Face:
    """50x50 px box: below the enroll/verify 60px floor, with usable landmarks
    so compare can still align and embed after the lenient quality gate."""
    return Face(
        box=(20, 20, 70, 70),
        score=0.97,
        landmarks=np.array(
            [
                [30.0, 35.0],
                [55.0, 35.0],
                [42.0, 48.0],
                [32.0, 58.0],
                [54.0, 58.0],
            ],
            dtype=np.float32,
        ),
    )


def _flat_image(size: int = 300) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_compare_accepts_small_face_on_image1(rigged_client):
    detector = rigged_client.app.state.recognizer.detector
    embedder = rigged_client.app.state.recognizer.embedder
    detector.queue([_too_small_face()], [_default_face()])
    base = _unit_vector(512, seed=1)
    embedder.queue(base, _nudged(base, seed=2))

    response = _compare(rigged_client)

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "Match"


def test_compare_accepts_small_face_on_image2(rigged_client):
    detector = rigged_client.app.state.recognizer.detector
    embedder = rigged_client.app.state.recognizer.embedder
    detector.queue([_default_face()], [_too_small_face()])
    base = _unit_vector(512, seed=1)
    embedder.queue(base, _nudged(base, seed=2))

    response = _compare(rigged_client)

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "Match"


def test_compare_small_face_on_image1_still_reports_no_face_on_image2(rigged_client):
    detector = rigged_client.app.state.recognizer.detector
    embedder = rigged_client.app.state.recognizer.embedder
    detector.queue([_too_small_face()], [])
    embedder.queue(_unit_vector(512, seed=1))

    response = _compare(rigged_client)

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "no_face_detected"
    assert body["details"]["failed_images"] == ["image2"]
    assert body["details"]["image1"] == {"ok": True}
    assert body["details"]["image2"]["error_code"] == "no_face_detected"
    assert "image1: OK" in body["message"]
    assert "image2:" in body["message"]


def test_compare_rejects_extremely_blurry_image2(rigged_client):
    detector = rigged_client.app.state.recognizer.detector
    embedder = rigged_client.app.state.recognizer.embedder
    detector.queue([_default_face()], [_default_face()])
    embedder.queue(_unit_vector(512, seed=1))

    response = rigged_client.post(
        "/api/v1/faces/compare",
        files={
            "image1": ("a.png", _encode_png(_valid_face_image()), "image/png"),
            "image2": ("b.png", _encode_png(_flat_image()), "image/png"),
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "low_image_quality"
    assert body["details"]["failed_images"] == ["image2"]
    assert body["details"]["image1"] == {"ok": True}
    assert "image_too_blurry" in body["details"]["image2"]["reasons"]
    assert "image1: OK" in body["message"]
    assert "too blurry" in body["message"]


def test_compare_rejects_extremely_blurry_on_both_images(rigged_client):
    rigged_client.app.state.recognizer.detector.faces_to_return = [_default_face()]
    blurry = _encode_png(_flat_image())

    response = rigged_client.post(
        "/api/v1/faces/compare",
        files={
            "image1": ("a.png", blurry, "image/png"),
            "image2": ("b.png", blurry, "image/png"),
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "low_image_quality"
    assert body["details"]["failed_images"] == ["image1", "image2"]
    assert "image_too_blurry" in body["details"]["image1"]["reasons"]
    assert "image_too_blurry" in body["details"]["image2"]["reasons"]
    assert "image1:" in body["message"]
    assert "image2:" in body["message"]


def test_enroll_still_rejects_face_too_small(rigged_client):
    rigged_client.app.state.recognizer.detector.faces_to_return = [_too_small_face()]

    response = _enroll(rigged_client, "EMP001")

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "low_image_quality"
    assert "face_too_small" in body["details"]["reasons"]


def test_compare_from_s3_urls(rigged_client, monkeypatch):
    image_bytes = _encode_png(_valid_face_image())

    async def fake_fetch(url, settings, **kwargs):
        assert "X-Amz-Signature" in url
        return image_bytes

    monkeypatch.setattr("app.api.remote_image.fetch_remote_image", fake_fetch)

    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base, _nudged(base, seed=2))

    response = rigged_client.post(
        "/api/v1/faces/compare",
        data={
            "image1_url": (
                "https://my-bucket.s3.ap-south-1.amazonaws.com/a.jpg?X-Amz-Signature=abc"
            ),
            "image2_url": (
                "https://my-bucket.s3.ap-south-1.amazonaws.com/b.jpg?X-Amz-Signature=def"
            ),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "Match"
    assert body["matched"] is True


def test_compare_mixed_file_and_s3_url(rigged_client, monkeypatch):
    image_bytes = _encode_png(_valid_face_image())

    async def fake_fetch(url, settings, **kwargs):
        return image_bytes

    monkeypatch.setattr("app.api.remote_image.fetch_remote_image", fake_fetch)

    embedder = rigged_client.app.state.recognizer.embedder
    base = _unit_vector(512, seed=1)
    embedder.queue(base, _nudged(base, seed=2))

    response = rigged_client.post(
        "/api/v1/faces/compare",
        data={"image2_url": "https://my-bucket.s3.ap-south-1.amazonaws.com/b.jpg"},
        files={"image1": ("a.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "Match"


def test_compare_missing_both_images_returns_400(client):
    response = client.post("/api/v1/faces/compare")

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image_source"


def test_verify_multi_rejects_invalid_frame_count(rigged_client):
    response = _verify_multi(rigged_client, num_frames=1, external_id="EMP001")

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_frame_count"
