from __future__ import annotations

import logging

import numpy as np

from app.core.exceptions import FaceServiceError, InconsistentImagesError, InvalidImageCountError
from app.core.recognizer import FaceEmbeddingResult, FaceRecognizer
from app.schemas.common import quality_result_to_schema
from app.schemas.embedding import EmbeddedImageSchema, EmbedResponse

logger = logging.getLogger(__name__)


def _annotate(exc: FaceServiceError, image_index: int) -> FaceServiceError:
    return type(exc)(
        f"image[{image_index}]: {exc.message}",
        image_index=image_index,
        **exc.details,
    )


class EmbeddingService:
    """Stateless embedding extraction: image(s) in, vectors out.

    Does not enroll, search FAISS, or persist anything. When more than one
    image is supplied, every later embedding must match the first (cosine
    similarity >= `consistency_threshold`) or the whole request fails.
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        consistency_threshold: float,
        max_images: int = 10,
    ) -> None:
        self.recognizer = recognizer
        self.consistency_threshold = consistency_threshold
        self.max_images = max_images

    def embed(self, images: list) -> EmbedResponse:
        image_count = len(images)
        if image_count < 1:
            raise InvalidImageCountError("At least one image is required", image_count=image_count)
        if image_count > self.max_images:
            raise InvalidImageCountError(
                f"At most {self.max_images} images are allowed per request",
                image_count=image_count,
                max_images=self.max_images,
            )

        results: list[FaceEmbeddingResult] = []
        for index, image in enumerate(images):
            try:
                results.append(self.recognizer.process(image, strict_single_face=True))
            except FaceServiceError as exc:
                raise _annotate(exc, index) from exc

        similarities: list[float | None] = [None]
        if image_count > 1:
            reference = results[0].embedding
            comparisons: list[dict] = []
            failed_indexes: list[int] = []
            for index, result in enumerate(results[1:], start=1):
                similarity = float(np.dot(reference, result.embedding))
                similarities.append(similarity)
                matched = similarity >= self.consistency_threshold
                comparisons.append({"index": index, "similarity": similarity, "matched": matched})
                if not matched:
                    failed_indexes.append(index)

            if failed_indexes:
                raise InconsistentImagesError(
                    "Images do not appear to show the same face",
                    threshold=self.consistency_threshold,
                    failed_indexes=failed_indexes,
                    comparisons=comparisons,
                )

        logger.info(
            "embed_completed",
            extra={
                "images_processed": image_count,
                "consistency_checked": image_count > 1,
            },
        )

        return EmbedResponse(
            images_processed=image_count,
            embeddings=[
                EmbeddedImageSchema(
                    index=index,
                    embedding=result.embedding.astype(float).tolist(),
                    detection_score=result.detection_score,
                    quality=quality_result_to_schema(result.quality),
                    similarity_to_first=similarities[index],
                )
                for index, result in enumerate(results)
            ],
            consistency_checked=image_count > 1,
            consistency_threshold=self.consistency_threshold if image_count > 1 else None,
        )
