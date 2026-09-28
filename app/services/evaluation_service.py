from __future__ import annotations

import numpy as np

from app.core.exceptions import (
    CompareImagesError,
    FaceServiceError,
    InvalidEmbeddingError,
    InvalidImageError,
    LowImageQualityError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.core.normalization import l2_normalize
from app.core.quality import QualityChecker, lenient_quality_thresholds
from app.core.recognizer import FaceEmbeddingResult, FaceRecognizer
from app.schemas.common import quality_result_to_schema
from app.schemas.verification import (
    FaceCompareEmbeddingResponse,
    FaceCompareImageSchema,
    FaceCompareResponse,
)

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
        # image2 (live capture): only extremely blurry crops fail.
        self.quality_checker = quality_checker or QualityChecker(lenient_quality_thresholds())
        # image1 / image1_url (stored/shared): no blur floor.
        self.image1_quality_checker = QualityChecker(lenient_quality_thresholds(min_sharpness=0.0))

    def _process_side(
        self,
        image,
        *,
        strict_single_face: bool,
        quality_checker: QualityChecker | None = None,
    ) -> tuple[FaceEmbeddingResult | None, FaceServiceError | None]:
        checker = quality_checker if quality_checker is not None else self.quality_checker
        try:
            return self.recognizer.process(
                image,
                strict_single_face=strict_single_face,
                quality_checker=checker,
            ), None
        except _COMPARE_SIDE_ERRORS as exc:
            return None, exc

    def compare_pair(self, image1, image2) -> FaceCompareResponse:
        """Primary two-image compare: exactly one face per image, no storage.

        Rejects ambiguous frames
        (`strict_single_face=True`) and returns Match / Not matching plus a
        confidence score instead of the legacy PASS/FAIL envelope.

        Quality is lenient on both sides (size/brightness/confidence never
        fail). Blur is checked only on image2 — image1 / image1_url is never
        rejected for blur. An extremely blurry capture, or no/multiple faces,
        still rejects. Both images are always processed so a failure names
        which side rejected.
        """
        result1, error1 = self._process_side(
            image1, strict_single_face=True, quality_checker=self.image1_quality_checker
        )
        result2, error2 = self._process_side(
            image2, strict_single_face=True, quality_checker=self.quality_checker
        )
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

    def compare_with_embedding(
        self,
        embedding: np.ndarray,
        image,
        *,
        expected_dimension: int,
    ) -> FaceCompareEmbeddingResponse:
        """Compare a caller-supplied embedding against one image.

        Same threshold and Match / Not matching outcome as `compare_pair`.
        The image uses the live-capture (image2) quality gate. The supplied
        vector is L2-normalized and must match `expected_dimension`.
        """
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != expected_dimension:
            raise InvalidEmbeddingError(
                f"embedding must be a 1-D vector of length {expected_dimension}",
                expected_dimension=expected_dimension,
                actual_dimension=int(vector.shape[0]) if vector.ndim == 1 else None,
                shape=list(vector.shape),
            )
        reference = l2_normalize(vector)

        result, error = self._process_side(
            image, strict_single_face=True, quality_checker=self.quality_checker
        )
        if error is not None:
            raise error
        assert result is not None

        confidence = float(np.dot(reference, result.embedding))
        matched = confidence >= self.similarity_threshold

        return FaceCompareEmbeddingResponse(
            matched=matched,
            status="Match" if matched else "Not matching",
            confidence=confidence,
            threshold=self.similarity_threshold,
            image=FaceCompareImageSchema(
                detection_score=result.detection_score,
                quality=quality_result_to_schema(result.quality),
            ),
        )
