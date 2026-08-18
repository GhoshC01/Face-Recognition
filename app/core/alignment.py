from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.exceptions import InvalidImageError

# Canonical 112x112 ArcFace landmark template (left-eye, right-eye, nose,
# left-mouth-corner, right-mouth-corner), in the same order SCRFD emits its
# 5-point landmarks. Any detector producing 5-point landmarks in this order
# can be aligned against it.
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_EXPECTED_LANDMARK_COUNT = 5


@dataclass
class AlignmentResult:
    """The standardized face crop plus metadata describing how it was produced."""

    aligned_image: np.ndarray
    transform_matrix: np.ndarray  # 2x3 similarity transform: source image -> aligned crop
    output_size: int
    landmarks: np.ndarray  # the validated (5, 2) landmarks used to compute the transform


def _validate_image(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise InvalidImageError("Face alignment received an empty or invalid image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"Face alignment requires an HxWx3 image, got shape {image.shape}")


def _validate_landmarks(landmarks: np.ndarray | None) -> np.ndarray:
    if landmarks is None:
        raise InvalidImageError("Face alignment requires landmarks, got None")

    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (_EXPECTED_LANDMARK_COUNT, 2):
        raise InvalidImageError(
            f"Face alignment requires {_EXPECTED_LANDMARK_COUNT} (x, y) landmarks, "
            f"got shape {landmarks.shape}"
        )

    if not np.all(np.isfinite(landmarks)):
        raise InvalidImageError("Face alignment received non-finite landmark coordinates")

    # A degenerate landmark set (e.g. all five points collapsed to one spot, as
    # a detector might emit for a failed/placeholder detection) has no unique
    # similarity-transform solution and must be rejected explicitly rather
    # than silently producing a garbage warp.
    spread = landmarks.max(axis=0) - landmarks.min(axis=0)
    if np.all(spread < 1e-3):
        raise InvalidImageError("Face alignment received degenerate (collapsed) landmarks")

    return landmarks


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> AlignmentResult:
    """Warp a detected face to a canonical frontal crop sized for the
    recognition model in use (MobileFaceNet expects 112x112, but `output_size`
    is left configurable so this module isn't tied to one specific model).

    A similarity transform (uniform scale + rotation + translation), rather
    than a full affine/perspective warp, is what ArcFace-family embedding
    models are trained on — it corrects in-plane rotation without distorting
    facial geometry.

    `image` is only read from (cv2.warpAffine writes into a freshly allocated
    output buffer), so the caller's original frame is never modified.
    """
    _validate_image(image)
    if output_size <= 0:
        raise InvalidImageError(f"output_size must be positive, got {output_size}")

    validated_landmarks = _validate_landmarks(landmarks)
    template = _ARCFACE_TEMPLATE * (output_size / 112.0)

    matrix, _ = cv2.estimateAffinePartial2D(validated_landmarks, template, method=cv2.LMEDS)
    if matrix is None:
        raise InvalidImageError("Could not estimate an alignment transform from the supplied landmarks")

    aligned = cv2.warpAffine(image, matrix, (output_size, output_size), borderValue=0.0)

    return AlignmentResult(
        aligned_image=aligned,
        transform_matrix=matrix,
        output_size=output_size,
        landmarks=validated_landmarks,
    )
