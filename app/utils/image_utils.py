from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import InvalidImageError


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR uint8 numpy array (OpenCV convention)."""
    if not data:
        raise InvalidImageError("Uploaded image is empty")

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise InvalidImageError("Could not decode image; unsupported or corrupted file")

    return image


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise InvalidImageError("Detected face box is degenerate")
    return image[y1:y2, x1:x2]
