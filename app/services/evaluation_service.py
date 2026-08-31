from __future__ import annotations

import numpy as np

from app.core.recognizer import FaceRecognizer
from app.schemas.common import quality_result_to_schema
from app.schemas.verification import CompareResponse, FaceCompareImageSchema, FaceCompareResponse


class EvaluationService:
    """Stateless 1:1 comparison of two images with no enrollment or storage
    involved. Useful for callers that just want "do these two faces match"
    without ever registering an identity in this service — e.g. ad-hoc
    verification, offline evaluation, or integration smoke tests.
    """

    def __init__(self, recognizer: FaceRecognizer, similarity_threshold: float) -> None:
        self.recognizer = recognizer
        self.similarity_threshold = similarity_threshold

    def compare(self, image_a, image_b) -> CompareResponse:
        result_a = self.recognizer.process(image_a, strict_single_face=False)
        result_b = self.recognizer.process(image_b, strict_single_face=False)

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
        """
        result1 = self.recognizer.process(image1, strict_single_face=True)
        result2 = self.recognizer.process(image2, strict_single_face=True)

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
