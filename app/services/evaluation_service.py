from __future__ import annotations

import numpy as np

from app.core.exceptions import (
    CompareImagesError,
    FaceServiceError,
    InvalidImageError,
    LowImageQualityError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.core.quality import QualityChecker, lenient_quality_thresholds
from app.core.recognizer import FaceEmbeddingResult, FaceRecognizer
from app.schemas.common import quality_result_to_schema
from app.schemas.verification import CompareResponse, FaceCompareImageSchema, FaceCompareResponse

# Per-image failures that should be attributed to image1/image2 rather than
# aborting the whole compare on the first side. Service-level errors
# (ModelNotReadyError, InvalidEmbeddingError) still propagate immediately.
_COMPARE_SIDE_ERRORS = (
    LowImageQualityError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    InvalidImageError,
)


def _side_payload(exc: FaceServiceError | None) -> dict:
    if exc is None:
        return {"ok": True}
    payload: dict = {
        "ok": False,
        "error_code": exc.error_code,
        "message": exc.message,
    }
    payload.update(exc.details)
    return payload


def raise_if_compare_sides_failed(
    error1: FaceServiceError | None,
    error2: FaceServiceError | None,
    *,
    label1: str = "image1",
    label2: str = "image2",
) -> None:
    """If either compare image failed, raise with both sides named.

    Same-reason failures keep that error_code (so `low_image_quality` stays
    `low_image_quality`). Mixed reasons become `compare_images_failed`.
    """
    if error1 is None and error2 is None:
        return

    failed = [name for name, err in ((label1, error1), (label2, error2)) if err is not None]
    details = {
        "failed_images": failed,
        label1: _side_payload(error1),
        label2: _side_payload(error2),
    }

    fragments = []
    for name, err in ((label1, error1), (label2, error2)):
        fragments.append(f"{name}: {err.message}" if err is not None else f"{name}: OK")
    combined = "; ".join(fragments)

    both_failed = error1 is not None and error2 is not None
    if both_failed and error1.error_code != error2.error_code:
        raise CompareImagesError(combined, **details)

    primary = error1 if error1 is not None else error2
    assert primary is not None
    extra = details if both_failed else {**primary.details, **details}
    raise type(primary)(combined, **extra)


class EvaluationService:
    """Stateless 1:1 comparison of two images with no enrollment or storage
    involved. Useful for callers that just want "do these two faces match"
    without ever registering an identity in this service — e.g. ad-hoc
    verification, offline evaluation, or integration smoke tests.
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        similarity_threshold: float,
        quality_checker: QualityChecker | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.similarity_threshold = similarity_threshold
        # Compare never uses the enroll/verify quality floor. Unset means the
        # built-in lenient preset (only extremely blurry crops fail).
        self.quality_checker = quality_checker or QualityChecker(lenient_quality_thresholds())

    def _process_side(
        self, image, *, strict_single_face: bool
    ) -> tuple[FaceEmbeddingResult | None, FaceServiceError | None]:
        try:
            return self.recognizer.process(
                image,
                strict_single_face=strict_single_face,
                quality_checker=self.quality_checker,
            ), None
        except _COMPARE_SIDE_ERRORS as exc:
            return None, exc

    def compare(self, image_a, image_b) -> CompareResponse:
        result_a, error_a = self._process_side(image_a, strict_single_face=False)
        result_b, error_b = self._process_side(image_b, strict_single_face=False)
        raise_if_compare_sides_failed(error_a, error_b, label1="file_a", label2="file_b")
        assert result_a is not None and result_b is not None

        similarity = float(np.dot(result_a.embedding, result_b.embedding))
        matched = similarity >= self.similarity_threshold

        return CompareResponse(
            match=matched,
            result="PASS" if matched else "FAIL",
            similarity_score=similarity,
            threshold=self.similarity_threshold,
        )

    def compare_pair(self, image1, image2) -> FaceCompareResponse:
        """Primary two-image compare: exactly one face per image, no storage.

        Same embedding pipeline as `compare()`, but rejects ambiguous frames
        (`strict_single_face=True`) and returns Match / Not matching plus a
        confidence score instead of the legacy PASS/FAIL envelope.

        Quality is lenient: size/brightness/confidence never fail a compare
        image. Only an extremely blurry crop (or no/multiple faces) rejects.
        Both images are always processed so a failure names *which* side
        rejected (image1 vs image2) and still reports the other side.
        """
        result1, error1 = self._process_side(image1, strict_single_face=True)
        result2, error2 = self._process_side(image2, strict_single_face=True)
        raise_if_compare_sides_failed(error1, error2)
        assert result1 is not None and result2 is not None

        confidence = float(np.dot(result1.embedding, result2.embedding))
        matched = confidence >= self.similarity_threshold

        return FaceCompareResponse(
            matched=matched,
            status="Match" if matched else "Not matching",
            confidence=confidence,
            threshold=self.similarity_threshold,
            image1=FaceCompareImageSchema(
                detection_score=result1.detection_score,
                quality=quality_result_to_schema(result1.quality),
            ),
            image2=FaceCompareImageSchema(
                detection_score=result2.detection_score,
                quality=quality_result_to_schema(result2.quality),
            ),
        )
