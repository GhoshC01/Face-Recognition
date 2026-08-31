from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.core.alignment import align_face
from app.core.detector import Face, FaceDetector
from app.core.embedding import FaceEmbedder
from app.core.exceptions import LowImageQualityError, MultipleFacesDetectedError, NoFaceDetectedError
from app.core.quality import QualityChecker, QualityResult, describe_quality_failure, quality_metrics_as_dict
from app.utils.timing import Stopwatch

logger = logging.getLogger(__name__)


@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray
    detection_score: float
    box: tuple[int, int, int, int]
    quality: QualityResult


class FaceRecognizer:
    """Orchestrates detection -> quality gating -> alignment -> embedding
    into a single 'image in, embedding out' pipeline."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        quality_checker: QualityChecker,
        max_faces_to_consider: int = 5,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.quality_checker = quality_checker
        self.max_faces_to_consider = max_faces_to_consider

    def _select_face(self, faces: list[Face], strict_single_face: bool) -> Face:
        if not faces:
            raise NoFaceDetectedError("No face detected in the supplied image")

        if strict_single_face and len(faces) > 1:
            raise MultipleFacesDetectedError(
                "Multiple faces detected; exactly one is required", face_count=len(faces)
            )

        return max(faces[: self.max_faces_to_consider], key=lambda f: f.area)

    def process(
        self,
        image: np.ndarray,
        strict_single_face: bool = False,
        quality_checker: QualityChecker | None = None,
    ) -> FaceEmbeddingResult:
        checker = quality_checker or self.quality_checker
        sw = Stopwatch()
        faces = self.detector.detect(image)
        detect_ms = sw.lap_ms()

        face = self._select_face(faces, strict_single_face)

        quality_result = checker.evaluate(image, face.box, face.score)
        quality_ms = sw.lap_ms()
        if not quality_result.accepted:
            raise LowImageQualityError(
                describe_quality_failure(quality_result, checker.thresholds),
                reasons=quality_result.reasons,
                quality_score=quality_result.quality_score,
                metrics=quality_metrics_as_dict(quality_result.metrics),
            )

        alignment = align_face(image, face.landmarks, output_size=self.embedder.input_size[0])
        align_ms = sw.lap_ms()
        embedding = self.embedder.embed(alignment.aligned_image)
        embed_ms = sw.lap_ms()

        logger.info(
            "recognizer_stage_timings",
            extra={
                "detect_ms": detect_ms,
                "quality_ms": quality_ms,
                "align_ms": align_ms,
                "embed_ms": embed_ms,
                "total_ms": round(detect_ms + quality_ms + align_ms + embed_ms, 2),
                "faces_detected": len(faces),
            },
        )

        return FaceEmbeddingResult(
            embedding=embedding,
            detection_score=face.score,
            box=face.box,
            quality=quality_result,
        )
