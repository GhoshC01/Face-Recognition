from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from app.core.exceptions import IdentityAlreadyExistsError, IdentityNotFoundError, InconsistentEnrollmentImagesError
from app.core.recognizer import FaceRecognizer
from app.core.vector_store import VectorStore
from app.schemas.common import quality_result_to_schema
from app.schemas.enrollment import (
    DualImageEnrollmentResponse,
    EnrolledImageInfo,
    EnrollmentDeleteResponse,
    EnrollmentResponse,
    EnrollmentStatusResponse,
)

logger = logging.getLogger(__name__)


class EnrollmentService:
    """Use case: register face image(s) against an external_id supplied by the
    caller. This service has no notion of "employee" — external_id is an
    opaque key chosen entirely by whichever system calls this API — and it
    never touches attendance data.
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        vector_store: VectorStore,
        duplicate_policy: str = "reject",
        min_image_similarity: float = 0.40,
        images_dir: str | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.vector_store = vector_store
        self.duplicate_policy = duplicate_policy
        self.min_image_similarity = min_image_similarity
        # Opt-in: unset (the default here) preserves the old embeddings-only
        # behavior relied on by existing callers/tests. app/main.py wires this
        # from settings.enrollment_images_dir for the running service.
        self.images_dir = images_dir or None

    def _save_image(self, external_id: str, embedding_id: int, image: np.ndarray) -> None:
        """Best-effort persistence of the raw enrollment photo for operator/audit
        reference. Never raises -- a disk failure here must not undo an
        already-successful embedding enrollment, since the FAISS write is the
        source of truth, not this copy."""
        if not self.images_dir:
            return
        try:
            folder = os.path.join(self.images_dir, external_id)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{embedding_id}.jpg")
            if not cv2.imwrite(path, image):
                raise OSError(f"cv2.imwrite returned False for {path}")
        except Exception:
            logger.warning(
                "enrollment_image_save_failed",
                extra={"external_id": external_id, "embedding_id": embedding_id},
                exc_info=True,
            )

    def enroll(self, external_id: str, image) -> EnrollmentResponse:
        result = self.recognizer.process(image, strict_single_face=True)
        embedding_id = self.vector_store.add_embedding(external_id, result.embedding)
        self._save_image(external_id, embedding_id, image)

        return EnrollmentResponse(
            external_id=external_id,
            enrolled=True,
            embedding_id=embedding_id,
            detection_score=result.detection_score,
            quality=quality_result_to_schema(result.quality),
        )

    def enroll_pair(self, external_id: str, image1, image2) -> DualImageEnrollmentResponse:
        """Initial enrollment workflow: exactly two images of the same person,
        each required to contain exactly one face. Nothing is written to the
        vector store until both images are fully validated (detected, quality
        gated, and cross-checked for consistency with each other) — so a
        rejected image never leaves a partial enrollment behind. The only
        remaining failure window is the storage step itself (e.g. the second
        FAISS write failing after the first succeeded), which is covered by
        an explicit rollback below.
        """
        is_duplicate = self.vector_store.has_identity(external_id)
        if is_duplicate and self.duplicate_policy == "reject":
            raise IdentityAlreadyExistsError(
                f"external_id='{external_id}' is already enrolled; re-enrollment is disabled by policy"
            )

        result1 = self.recognizer.process(image1, strict_single_face=True)
        result2 = self.recognizer.process(image2, strict_single_face=True)

        image_similarity = float(np.dot(result1.embedding, result2.embedding))
        if image_similarity < self.min_image_similarity:
            raise InconsistentEnrollmentImagesError(
                "The two enrollment images do not appear to show the same person",
                similarity_score=image_similarity,
                threshold=self.min_image_similarity,
            )

        # Both images are known-good and mutually consistent -- only now do we
        # touch stored state. Under the "replace" policy, the previous
        # enrollment is cleared here (not earlier), so a rejected pair never
        # destroys a still-valid prior enrollment.
        if is_duplicate:
            self.vector_store.remove_embedding(external_id)

        embedding_id_1 = self.vector_store.add_embedding(external_id, result1.embedding)
        try:
            embedding_id_2 = self.vector_store.add_embedding(external_id, result2.embedding)
        except Exception:
            self.vector_store.remove_embedding(external_id)
            logger.warning(
                "enrollment_rolled_back",
                extra={"external_id": external_id, "reason": "second_image_storage_failed"},
            )
            raise

        self._save_image(external_id, embedding_id_1, image1)
        self._save_image(external_id, embedding_id_2, image2)

        return DualImageEnrollmentResponse(
            external_id=external_id,
            images_processed=2,
            image_similarity=image_similarity,
            images=[
                EnrolledImageInfo(
                    image="image1",
                    embedding_id=embedding_id_1,
                    detection_score=result1.detection_score,
                    quality=quality_result_to_schema(result1.quality),
                ),
                EnrolledImageInfo(
                    image="image2",
                    embedding_id=embedding_id_2,
                    detection_score=result2.detection_score,
                    quality=quality_result_to_schema(result2.quality),
                ),
            ],
        )

    def status(self, external_id: str) -> EnrollmentStatusResponse:
        embeddings = self.vector_store.get_embeddings(external_id)
        return EnrollmentStatusResponse(
            external_id=external_id,
            enrolled=len(embeddings) > 0,
            embedding_count=len(embeddings),
        )

    def remove(self, external_id: str) -> EnrollmentDeleteResponse:
        if not self.vector_store.has_identity(external_id):
            raise IdentityNotFoundError(f"No enrolled embeddings found for external_id='{external_id}'")

        removed_count = self.vector_store.remove_embedding(external_id)
        self._remove_images(external_id)
        return EnrollmentDeleteResponse(
            external_id=external_id,
            removed=removed_count > 0,
            embeddings_removed=removed_count,
        )

    def _remove_images(self, external_id: str) -> None:
        """Best-effort cleanup of an identity's saved photos on full removal --
        never raises, since the FAISS removal above is what actually matters
        for correctness."""
        if not self.images_dir:
            return
        try:
            folder = os.path.join(self.images_dir, external_id)
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    os.remove(os.path.join(folder, name))
                os.rmdir(folder)
        except OSError:
            logger.warning("enrollment_image_cleanup_failed", extra={"external_id": external_id}, exc_info=True)
