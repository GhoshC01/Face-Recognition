from __future__ import annotations

import numpy as np

from app.core.recognizer import FaceRecognizer
from app.schemas.verification import CompareResponse


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
