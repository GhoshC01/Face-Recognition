from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from app.core.exceptions import IdentityAlreadyExistsError
from app.core.recognizer import FaceRecognizer
from app.core.vector_store import VectorStore
from app.schemas.common import quality_result_to_schema
from app.schemas.enrollment import EnrollmentResponse
from app.utils.timing import Stopwatch

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
        images_dir: str | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.vector_store = vector_store
        self.duplicate_policy = duplicate_policy
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

    def enroll_initial(self, external_id: str, image) -> EnrollmentResponse:
        """Initial enrollment workflow: a single image of the person, required
        to contain exactly one face and pass quality checks. Nothing is
        written to the vector store until the image is fully validated -- so
        a rejected image never leaves a partial enrollment behind.
        """
        sw = Stopwatch()
        is_duplicate = self.vector_store.has_identity(external_id)
        if is_duplicate and self.duplicate_policy == "reject":
            raise IdentityAlreadyExistsError(
                f"external_id='{external_id}' is already enrolled; re-enrollment is disabled by policy"
            )

        result = self.recognizer.process(image, strict_single_face=True)
        recognize_ms = sw.lap_ms()

        # The image is known-good -- only now do we touch stored state. Under
        # the "replace" policy, the previous enrollment is cleared here (not
        # earlier), so a rejected image never destroys a still-valid prior
        # enrollment.
        if is_duplicate:
            self.vector_store.remove_embedding(external_id)

        embedding_id = self.vector_store.add_embedding(external_id, result.embedding)
        storage_ms = sw.lap_ms()

        self._save_image(external_id, embedding_id, image)
        image_save_ms = sw.lap_ms()

        logger.info(
            "enroll_initial_stage_timings",
            extra={
                "external_id": external_id,
                "recognize_ms": recognize_ms,
                "vector_store_write_ms": storage_ms,
                "image_save_ms": image_save_ms,
                "total_ms": round(recognize_ms + storage_ms + image_save_ms, 2),
            },
        )

        return EnrollmentResponse(
            external_id=external_id,
            enrolled=True,
            embedding_id=embedding_id,
            detection_score=result.detection_score,
            quality=quality_result_to_schema(result.quality),
        )
