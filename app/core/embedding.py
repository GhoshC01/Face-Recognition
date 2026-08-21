from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.exceptions import InvalidImageError, ModelNotReadyError
from app.core.normalization import l2_normalize
from app.utils.timing import Stopwatch

logger = logging.getLogger(__name__)


def _infer_embedding_dimension(output_shape) -> int:
    """Reads the embedding length out of an ONNX output tensor's declared
    shape, e.g. [1, 512] or ['batch', 512] -> 512. Kept separate from
    `FaceEmbedder.load()` so this parsing logic is unit-testable without a
    real ONNX Runtime session."""
    if not output_shape:
        raise ModelNotReadyError(f"Recognizer model output shape is empty: {output_shape!r}")

    dimension = output_shape[-1]
    if not isinstance(dimension, int) or dimension <= 0:
        raise ModelNotReadyError(
            f"Recognizer model declares a non-fixed embedding dimension in its "
            f"output shape {output_shape!r}; expected a fixed positive integer "
            "in the last axis"
        )
    return dimension


class FaceEmbedder:
    """ONNX Runtime wrapper for an ArcFace-family embedding model
    (e.g. w600k_mbf.onnx / MobileFaceNet). This component does exactly one
    thing: aligned face in, L2-normalized embedding vector out. It has no
    knowledge of FAISS, employees, or attendance, and never identifies
    anyone -- it only produces a numeric vector for the caller to compare.

    Embeddings are L2-normalized so cosine similarity reduces to a plain dot
    product, matching FAISS IndexFlatIP.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ) -> None:
        self.model_path = model_path
        self.input_size = (input_size, input_size)
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._embedding_dimension: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def embedding_dimension(self) -> int | None:
        """The embedding vector length, read from the loaded model's own
        output metadata -- never assumed ahead of time. `None` until `load()`
        has run successfully."""
        return self._embedding_dimension

    def load(self) -> None:
        if self._session is not None:
            return

        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Recognizer model not found at '{path}'")

        sw = Stopwatch()
        options = ort.SessionOptions()
        if self._intra_op_threads:
            options.intra_op_num_threads = self._intra_op_threads
        if self._inter_op_threads:
            options.inter_op_num_threads = self._inter_op_threads

        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

        # The embedding dimension is whatever the model's own output signature
        # says it is -- it is read here, once, rather than hardcoded, so a
        # different export (e.g. a 256-d or 128-d variant) is picked up
        # automatically instead of silently truncating/misreading output.
        dimension = _infer_embedding_dimension(session.get_outputs()[0].shape)

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._embedding_dimension = dimension
        logger.info(
            "recognizer_loaded",
            extra={"model_path": str(path), "embedding_dimension": dimension, "load_ms": sw.lap_ms()},
        )

    @staticmethod
    def _validate_aligned_face(aligned_face: np.ndarray) -> None:
        if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
            raise InvalidImageError("Embedder received an empty or invalid aligned face")
        if aligned_face.ndim != 3 or aligned_face.shape[2] != 3:
            raise InvalidImageError(
                f"Embedder requires an HxWx3 aligned face, got shape {aligned_face.shape}"
            )

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        self._validate_aligned_face(aligned_face)

        if self._session is None:
            logger.warning("embedder_lazy_load_triggered_by_request")
            try:
                self.load()
            except FileNotFoundError as exc:
                raise ModelNotReadyError("Face embedding model is not available") from exc
        assert self._session is not None

        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        output = self._session.run(None, {self._input_name: blob})[0]
        embedding = output.reshape(-1).astype(np.float32)

        if embedding.shape[0] != self._embedding_dimension:
            raise ModelNotReadyError(
                f"Recognizer model produced an embedding of length {embedding.shape[0]}, "
                f"expected {self._embedding_dimension} based on its declared output shape"
            )

        return l2_normalize(embedding)
