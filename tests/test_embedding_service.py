from __future__ import annotations

import pytest

from app.core.exceptions import (
    InconsistentImagesError,
    InvalidImageCountError,
    NoFaceDetectedError,
)
from app.services.embedding_service import EmbeddingService
from tests.conftest import FakeRecognizer, make_synthetic_image


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer(dimension=8)


@pytest.fixture
def service(recognizer) -> EmbeddingService:
    return EmbeddingService(recognizer=recognizer, consistency_threshold=0.40, max_images=5)


def test_embed_single_image_returns_vector_without_consistency_check(service):
    response = service.embed([make_synthetic_image(7)])

    assert response.images_processed == 1
    assert response.consistency_checked is False
    assert response.consistency_threshold is None
    assert len(response.embeddings) == 1
    item = response.embeddings[0]
    assert item.index == 0
    assert len(item.embedding) == 8
    assert item.similarity_to_first is None
    assert item.detection_score == 0.99


def test_embed_multiple_same_face_returns_all_embeddings(service):
    response = service.embed(
        [make_synthetic_image(7), make_synthetic_image(7), make_synthetic_image(7)]
    )

    assert response.images_processed == 3
    assert response.consistency_checked is True
    assert response.consistency_threshold == 0.40
    assert [item.index for item in response.embeddings] == [0, 1, 2]
    assert response.embeddings[0].similarity_to_first is None
    assert response.embeddings[1].similarity_to_first == pytest.approx(1.0)
    assert response.embeddings[2].similarity_to_first == pytest.approx(1.0)
    assert response.embeddings[0].embedding == response.embeddings[1].embedding


def test_embed_rejects_when_a_later_image_is_a_different_face(service):
    with pytest.raises(InconsistentImagesError) as exc_info:
        service.embed(
            [make_synthetic_image(7), make_synthetic_image(7), make_synthetic_image(42)]
        )

    details = exc_info.value.details
    assert exc_info.value.error_code == "inconsistent_images"
    assert details["failed_indexes"] == [2]
    assert details["threshold"] == 0.40
    assert details["comparisons"][0]["index"] == 1
    assert details["comparisons"][0]["matched"] is True
    assert details["comparisons"][1]["index"] == 2
    assert details["comparisons"][1]["matched"] is False


def test_embed_only_compares_later_images_against_the_first(service):
    # image[1] and image[2] can differ from each other as long as both match image[0].
    # FakeRecognizer: same marker => identical embedding, so this is the same-person case.
    response = service.embed([make_synthetic_image(3), make_synthetic_image(3)])
    assert response.consistency_checked is True
    assert response.embeddings[1].similarity_to_first >= 0.40


def test_embed_rejects_empty_list(service):
    with pytest.raises(InvalidImageCountError) as exc_info:
        service.embed([])
    assert exc_info.value.details["image_count"] == 0


def test_embed_rejects_more_images_than_max(recognizer):
    service = EmbeddingService(recognizer=recognizer, consistency_threshold=0.40, max_images=2)

    with pytest.raises(InvalidImageCountError) as exc_info:
        service.embed([make_synthetic_image(1), make_synthetic_image(1), make_synthetic_image(1)])

    assert exc_info.value.details["max_images"] == 2
    assert exc_info.value.details["image_count"] == 3


def test_embed_annotates_per_image_pipeline_errors():
    class _FailingRecognizer(FakeRecognizer):
        def process(self, image, strict_single_face=False, quality_checker=None):
            marker = int(image[0, 0, 0])
            if marker == 9:
                raise NoFaceDetectedError("No face detected in the supplied image")
            return super().process(image, strict_single_face=strict_single_face)

    service = EmbeddingService(_FailingRecognizer(), consistency_threshold=0.40)

    with pytest.raises(NoFaceDetectedError) as exc_info:
        service.embed([make_synthetic_image(7), make_synthetic_image(9)])

    assert "image[1]:" in exc_info.value.message
    assert exc_info.value.details["image_index"] == 1
