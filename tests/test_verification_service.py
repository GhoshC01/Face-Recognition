from __future__ import annotations

import pytest

from app.core.exceptions import IdentityNotFoundError
from app.core.vector_store import VectorStore
from app.services.enrollment_service import EnrollmentService
from app.services.evaluation_service import EvaluationService
from app.services.verification_service import VerificationService
from tests.conftest import FakeRecognizer, make_synthetic_image


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(8, str(tmp_path / "faiss"), str(tmp_path / "metadata"), "index.faiss", "metadata.json")


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer(dimension=8)


def test_enroll_then_verify_passes(store, recognizer):
    enrollment = EnrollmentService(recognizer=recognizer, vector_store=store)
    verification = VerificationService(
        recognizer=recognizer,
        vector_store=store,
        verification_threshold=0.99,
        identification_threshold=0.99,
        identification_top_k=5,
    )

    enrollment.enroll("emp-1", make_synthetic_image(marker=7))
    result = verification.verify("emp-1", make_synthetic_image(marker=7))

    assert result.result == "PASS"
    assert result.verified is True


def test_verify_fails_for_different_face(store, recognizer):
    enrollment = EnrollmentService(recognizer=recognizer, vector_store=store)
    verification = VerificationService(
        recognizer=recognizer,
        vector_store=store,
        verification_threshold=0.99,
        identification_threshold=0.99,
        identification_top_k=5,
    )

    enrollment.enroll("emp-1", make_synthetic_image(marker=7))
    result = verification.verify("emp-1", make_synthetic_image(marker=99))

    assert result.result == "FAIL"
    assert result.verified is False


def test_verify_unknown_identity_raises(store, recognizer):
    verification = VerificationService(
        recognizer=recognizer,
        vector_store=store,
        verification_threshold=0.99,
        identification_threshold=0.99,
        identification_top_k=5,
    )

    with pytest.raises(IdentityNotFoundError):
        verification.verify("ghost", make_synthetic_image(marker=1))


def test_evaluation_service_compares_two_images_directly(recognizer):
    evaluation = EvaluationService(recognizer=recognizer, similarity_threshold=0.99)

    same_face = evaluation.compare(make_synthetic_image(7), make_synthetic_image(7))
    different_face = evaluation.compare(make_synthetic_image(7), make_synthetic_image(42))

    assert same_face.result == "PASS"
    assert different_face.result == "FAIL"


def test_enrollment_remove_unknown_identity_raises(store, recognizer):
    enrollment = EnrollmentService(recognizer=recognizer, vector_store=store)
    with pytest.raises(IdentityNotFoundError):
        enrollment.remove("ghost")
