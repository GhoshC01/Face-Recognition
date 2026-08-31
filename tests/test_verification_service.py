from __future__ import annotations

import pytest

from app.core.exceptions import (
    CompareImagesError,
    IdentityNotFoundError,
    LowImageQualityError,
    NoFaceDetectedError,
)
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


class _SideAwareRecognizer(FakeRecognizer):
    """Runs FakeRecognizer unless a given call index is mapped to an exception.

    Call 0 is image1 / file_a; call 1 is image2 / file_b.
    """

    def __init__(self, errors_by_call: dict[int, Exception], dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self._call = 0
        self._errors = errors_by_call

    def process(self, image, strict_single_face: bool = False, quality_checker=None):
        idx = self._call
        self._call += 1
        if idx in self._errors:
            raise self._errors[idx]
        return super().process(image, strict_single_face=strict_single_face, quality_checker=quality_checker)


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


def test_evaluation_service_compare_pair_returns_match_and_confidence(recognizer):
    evaluation = EvaluationService(recognizer=recognizer, similarity_threshold=0.99)

    same_face = evaluation.compare_pair(make_synthetic_image(7), make_synthetic_image(7))
    different_face = evaluation.compare_pair(make_synthetic_image(7), make_synthetic_image(42))

    assert same_face.status == "Match"
    assert same_face.matched is True
    assert same_face.confidence >= same_face.threshold

    assert different_face.status == "Not matching"
    assert different_face.matched is False
    assert different_face.confidence < different_face.threshold


def test_compare_pair_names_image1_when_only_first_fails():
    recognizer = _SideAwareRecognizer(
        {
            0: LowImageQualityError(
                "Face image did not pass quality checks: image is too blurry (sharpness 1; minimum 8)",
                reasons=["image_too_blurry"],
                quality_score=0.20,
            )
        }
    )
    evaluation = EvaluationService(recognizer=recognizer, similarity_threshold=0.99)

    with pytest.raises(LowImageQualityError) as caught:
        evaluation.compare_pair(make_synthetic_image(7), make_synthetic_image(7))

    err = caught.value
    assert err.error_code == "low_image_quality"
    assert err.details["failed_images"] == ["image1"]
    assert err.details["image1"]["ok"] is False
    assert err.details["image1"]["reasons"] == ["image_too_blurry"]
    assert err.details["image2"] == {"ok": True}
    assert "image1:" in err.message
    assert "image2: OK" in err.message
    assert "too blurry" in err.message


def test_compare_pair_names_image2_when_only_second_fails():
    recognizer = _SideAwareRecognizer(
        {
            1: LowImageQualityError(
                "Face image did not pass quality checks: image is too blurry (sharpness 1; minimum 8)",
                reasons=["image_too_blurry"],
                quality_score=0.20,
            )
        }
    )
    evaluation = EvaluationService(recognizer=recognizer, similarity_threshold=0.99)

    with pytest.raises(LowImageQualityError) as caught:
        evaluation.compare_pair(make_synthetic_image(7), make_synthetic_image(42))

    err = caught.value
    assert err.details["failed_images"] == ["image2"]
    assert err.details["image1"] == {"ok": True}
    assert err.details["image2"]["ok"] is False
    assert err.details["image2"]["reasons"] == ["image_too_blurry"]
    assert "image1: OK" in err.message
    assert "image2:" in err.message


def test_compare_pair_reports_both_images_when_both_fail():
    recognizer = _SideAwareRecognizer(
        {
            0: LowImageQualityError(
                "Face image did not pass quality checks: image is too blurry (sharpness 0; minimum 8)",
                reasons=["image_too_blurry"],
                quality_score=0.10,
            ),
            1: NoFaceDetectedError("No face detected in the supplied image"),
        }
    )
    evaluation = EvaluationService(recognizer=recognizer, similarity_threshold=0.99)

    with pytest.raises(CompareImagesError) as caught:
        evaluation.compare_pair(make_synthetic_image(7), make_synthetic_image(42))

    err = caught.value
    assert err.error_code == "compare_images_failed"
    assert err.details["failed_images"] == ["image1", "image2"]
    assert err.details["image1"]["error_code"] == "low_image_quality"
    assert err.details["image2"]["error_code"] == "no_face_detected"
    assert "image1:" in err.message
    assert "image2:" in err.message
    assert "OK" not in err.message


def test_enrollment_remove_unknown_identity_raises(store, recognizer):
    enrollment = EnrollmentService(recognizer=recognizer, vector_store=store)
    with pytest.raises(IdentityNotFoundError):
        enrollment.remove("ghost")
