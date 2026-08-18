from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import (
    IdentityAlreadyExistsError,
    InconsistentEnrollmentImagesError,
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


def _nudged(base: np.ndarray, seed: int, noise_scale: float = 0.05) -> np.ndarray:
    """A vector close to `base` (simulates a second photo of the same person)."""
    rng = np.random.default_rng(seed)
    nudged = base + rng.normal(scale=noise_scale, size=base.shape).astype(np.float32)
    return nudged / np.linalg.norm(nudged)


def _orthogonal(dimension: int = DIMENSION) -> np.ndarray:
    vec = np.zeros(dimension, dtype=np.float32)
    vec[1] = 1.0
    return vec


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
    raise -- so enroll_pair's branching can be tested deterministically
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


class FlakyVectorStore(VectorStore):
    """A VectorStore whose add_embedding fails starting from the Nth call,
    to simulate a storage failure partway through a dual-image enrollment."""

    def __init__(self, *args, fail_on_call: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._call_count = 0
        self._fail_on_call = fail_on_call

    def add_embedding(self, external_id: str, embedding: np.ndarray) -> int:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError("simulated storage failure")
        return super().add_embedding(external_id, embedding)


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(DIMENSION, str(tmp_path / "faiss"), str(tmp_path / "metadata"), "index.faiss", "metadata.json")


# --- success path ---


def test_enroll_pair_success(store):
    emb_a = _unit_vector(1)
    emb_b = _nudged(emb_a, seed=2)
    recognizer = ScriptedRecognizer({1: _result(emb_a), 2: _result(emb_b)})
    service = EnrollmentService(recognizer, store, duplicate_policy="reject", min_image_similarity=0.40)

    response = service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert response.success is True
    assert response.external_id == "EMP001"
    assert response.images_processed == 2
    assert response.enrollment_status == "success"
    assert len(response.images) == 2
    assert {img.image for img in response.images} == {"image1", "image2"}
    assert store.count() == 2
    assert store.has_identity("EMP001")


# --- one face per image ---


def test_enroll_pair_rejects_no_face_image(store):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1)), 2: NoFaceDetectedError("no face")})
    service = EnrollmentService(recognizer, store)

    with pytest.raises(NoFaceDetectedError):
        service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert store.count() == 0
    assert not store.has_identity("EMP001")


def test_enroll_pair_rejects_multiple_face_image(store):
    recognizer = ScriptedRecognizer(
        {1: _result(_unit_vector(1)), 2: MultipleFacesDetectedError("2 faces", face_count=2)}
    )
    service = EnrollmentService(recognizer, store)

    with pytest.raises(MultipleFacesDetectedError):
        service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert store.count() == 0


# --- cross-image consistency check ---


def test_enroll_pair_rejects_inconsistent_images(store):
    emb_a = _unit_vector(1)
    emb_dissimilar = _orthogonal()
    recognizer = ScriptedRecognizer({1: _result(emb_a), 2: _result(emb_dissimilar)})
    service = EnrollmentService(recognizer, store, min_image_similarity=0.40)

    with pytest.raises(InconsistentEnrollmentImagesError):
        service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert store.count() == 0
    assert not store.has_identity("EMP001")


# --- duplicate enrollment policy ---


def test_duplicate_enrollment_rejected_by_default(store):
    store.add_embedding("EMP001", _unit_vector(1))
    recognizer = ScriptedRecognizer({2: _result(_unit_vector(2)), 3: _result(_unit_vector(3))})
    service = EnrollmentService(recognizer, store, duplicate_policy="reject")

    with pytest.raises(IdentityAlreadyExistsError):
        service.enroll_pair("EMP001", make_synthetic_image(2), make_synthetic_image(3))

    assert recognizer.calls == []  # rejected before any image processing
    assert store.count() == 1  # original enrollment untouched


def test_duplicate_enrollment_replaced_when_policy_allows(store):
    old_embedding = _unit_vector(99)
    store.add_embedding("EMP001", old_embedding)

    emb_a = _unit_vector(1)
    emb_b = _nudged(emb_a, seed=2)
    recognizer = ScriptedRecognizer({1: _result(emb_a), 2: _result(emb_b)})
    service = EnrollmentService(recognizer, store, duplicate_policy="replace", min_image_similarity=0.40)

    response = service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert response.enrollment_status == "success"
    assert store.count() == 2  # old embedding replaced, not appended to
    stored = store.get_embeddings("EMP001")
    for vec in stored:
        assert not np.allclose(vec, old_embedding, atol=1e-4)


def test_replace_policy_does_not_destroy_old_enrollment_on_validation_failure(store):
    """A rejected re-enrollment attempt must never cost the caller their
    previously valid enrollment."""
    old_embedding = _unit_vector(99)
    store.add_embedding("EMP001", old_embedding)

    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1)), 2: NoFaceDetectedError("no face")})
    service = EnrollmentService(recognizer, store, duplicate_policy="replace")

    with pytest.raises(NoFaceDetectedError):
        service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert store.count() == 1
    np.testing.assert_allclose(store.get_embeddings("EMP001")[0], old_embedding, atol=1e-5)


# --- rollback on partial storage failure ---


def test_rolls_back_first_embedding_when_second_storage_write_fails(tmp_path):
    flaky_store = FlakyVectorStore(
        DIMENSION,
        str(tmp_path / "faiss"),
        str(tmp_path / "metadata"),
        "index.faiss",
        "metadata.json",
        fail_on_call=2,
    )
    emb_a = _unit_vector(1)
    emb_b = _nudged(emb_a, seed=2)
    recognizer = ScriptedRecognizer({1: _result(emb_a), 2: _result(emb_b)})
    service = EnrollmentService(recognizer, flaky_store, min_image_similarity=0.40)

    with pytest.raises(RuntimeError):
        service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    assert flaky_store.count() == 0
    assert not flaky_store.has_identity("EMP001")


# --- raw enrollment photo persistence (opt-in via images_dir) ---


def test_enroll_pair_saves_both_photos_when_images_dir_set(store, tmp_path):
    emb_a = _unit_vector(1)
    emb_b = _nudged(emb_a, seed=2)
    recognizer = ScriptedRecognizer({1: _result(emb_a), 2: _result(emb_b)})
    images_dir = tmp_path / "images"
    service = EnrollmentService(recognizer, store, min_image_similarity=0.40, images_dir=str(images_dir))

    response = service.enroll_pair("EMP001", make_synthetic_image(1), make_synthetic_image(2))

    saved = sorted(p.name for p in (images_dir / "EMP001").iterdir())
    assert saved == sorted(f"{img.embedding_id}.jpg" for img in response.images)


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
