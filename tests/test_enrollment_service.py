from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import (
    IdentityAlreadyExistsError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.core.quality import QualityObservedMetrics, QualityResult
from app.core.recognizer import FaceEmbeddingResult
from app.core.vector_store import VectorStore
from app.services.enrollment_service import EnrollmentService
from tests.conftest import make_synthetic_image

DIMENSION = 8


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=DIMENSION).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _result(embedding: np.ndarray, detection_score: float = 0.95) -> FaceEmbeddingResult:
    return FaceEmbeddingResult(
        embedding=embedding,
        detection_score=detection_score,
        box=(0, 0, 100, 100),
        quality=QualityResult(
            accepted=True,
            quality_score=0.9,
            reasons=[],
            metrics=QualityObservedMetrics(
                detection_confidence=detection_score,
                face_width=100,
                face_height=100,
                face_area_ratio=0.5,
                brightness=120.0,
                sharpness=150.0,
            ),
        ),
    )


class ScriptedRecognizer:
    """Maps a marker byte embedded in a synthetic image to a scripted outcome
    -- either a FaceEmbeddingResult to return, or an exception instance to
    raise -- so enroll_initial's branching can be tested deterministically
    without real ONNX models."""

    def __init__(self, outcomes: dict[int, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[int] = []

    def process(self, image: np.ndarray, strict_single_face: bool = False):
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        outcome = self.outcomes[marker]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(DIMENSION, str(tmp_path / "faiss"), str(tmp_path / "metadata"), "index.faiss", "metadata.json")


# --- success path ---


def test_enroll_initial_success(store):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    service = EnrollmentService(recognizer, store, duplicate_policy="reject")

    response = service.enroll_initial("EMP001", make_synthetic_image(1))

    assert response.enrolled is True
    assert response.external_id == "EMP001"
    assert store.count() == 1
    assert store.has_identity("EMP001")


# --- one face per image ---


def test_enroll_initial_rejects_no_face_image(store):
    recognizer = ScriptedRecognizer({1: NoFaceDetectedError("no face")})
    service = EnrollmentService(recognizer, store)

    with pytest.raises(NoFaceDetectedError):
        service.enroll_initial("EMP001", make_synthetic_image(1))

    assert store.count() == 0
    assert not store.has_identity("EMP001")


def test_enroll_initial_rejects_multiple_face_image(store):
    recognizer = ScriptedRecognizer({1: MultipleFacesDetectedError("2 faces", face_count=2)})
    service = EnrollmentService(recognizer, store)

    with pytest.raises(MultipleFacesDetectedError):
        service.enroll_initial("EMP001", make_synthetic_image(1))

    assert store.count() == 0


# --- duplicate enrollment policy ---


def test_duplicate_enrollment_rejected_by_default(store):
    store.add_embedding("EMP001", _unit_vector(1))
    recognizer = ScriptedRecognizer({2: _result(_unit_vector(2))})
    service = EnrollmentService(recognizer, store, duplicate_policy="reject")

    with pytest.raises(IdentityAlreadyExistsError):
        service.enroll_initial("EMP001", make_synthetic_image(2))

    assert recognizer.calls == []  # rejected before any image processing
    assert store.count() == 1  # original enrollment untouched


def test_duplicate_enrollment_replaced_when_policy_allows(store):
    old_embedding = _unit_vector(99)
    store.add_embedding("EMP001", old_embedding)

    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    service = EnrollmentService(recognizer, store, duplicate_policy="replace")

    response = service.enroll_initial("EMP001", make_synthetic_image(1))

    assert response.enrolled is True
    assert store.count() == 1  # old embedding replaced, not appended to
    stored = store.get_embeddings("EMP001")
    for vec in stored:
        assert not np.allclose(vec, old_embedding, atol=1e-4)


def test_replace_policy_does_not_destroy_old_enrollment_on_validation_failure(store):
    """A rejected re-enrollment attempt must never cost the caller their
    previously valid enrollment."""
    old_embedding = _unit_vector(99)
    store.add_embedding("EMP001", old_embedding)

    recognizer = ScriptedRecognizer({1: NoFaceDetectedError("no face")})
    service = EnrollmentService(recognizer, store, duplicate_policy="replace")

    with pytest.raises(NoFaceDetectedError):
        service.enroll_initial("EMP001", make_synthetic_image(1))

    assert store.count() == 1
    np.testing.assert_allclose(store.get_embeddings("EMP001")[0], old_embedding, atol=1e-5)


# --- raw enrollment photo persistence (opt-in via images_dir) ---


def test_enroll_initial_saves_photo_when_images_dir_set(store, tmp_path):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    images_dir = tmp_path / "images"
    service = EnrollmentService(recognizer, store, images_dir=str(images_dir))

    response = service.enroll_initial("EMP001", make_synthetic_image(1))

    assert (images_dir / "EMP001" / f"{response.embedding_id}.jpg").is_file()


def test_enroll_single_image_saves_photo_when_images_dir_set(store, tmp_path):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    images_dir = tmp_path / "images"
    service = EnrollmentService(recognizer, store, images_dir=str(images_dir))

    response = service.enroll("EMP001", make_synthetic_image(1))

    assert (images_dir / "EMP001" / f"{response.embedding_id}.jpg").is_file()


def test_no_images_saved_when_images_dir_not_set(store, tmp_path):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    service = EnrollmentService(recognizer, store)  # images_dir defaults to None

    service.enroll("EMP001", make_synthetic_image(1))

    assert not (tmp_path / "images").exists()


def test_remove_deletes_saved_photos(store, tmp_path):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    images_dir = tmp_path / "images"
    service = EnrollmentService(recognizer, store, images_dir=str(images_dir))
    service.enroll("EMP001", make_synthetic_image(1))
    assert (images_dir / "EMP001").is_dir()

    service.remove("EMP001")

    assert not (images_dir / "EMP001").exists()
