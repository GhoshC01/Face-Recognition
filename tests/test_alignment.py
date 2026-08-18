from __future__ import annotations

import numpy as np
import pytest

from app.core.alignment import _ARCFACE_TEMPLATE, AlignmentResult, align_face
from app.core.exceptions import InvalidImageError


def _shifted_landmarks(offset=(20.0, 30.0)) -> np.ndarray:
    return _ARCFACE_TEMPLATE + np.array(offset, dtype=np.float32)


def _random_image(size: int = 200) -> np.ndarray:
    return np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_align_face_returns_standardized_crop_and_metadata():
    image = _random_image()
    landmarks = _shifted_landmarks()

    result = align_face(image, landmarks, output_size=112)

    assert isinstance(result, AlignmentResult)
    assert result.aligned_image.shape == (112, 112, 3)
    assert result.aligned_image.dtype == np.uint8
    assert result.output_size == 112
    assert result.transform_matrix.shape == (2, 3)
    np.testing.assert_allclose(result.landmarks, landmarks)


def test_output_size_is_configurable_for_the_recognition_model():
    image = _random_image()
    landmarks = _shifted_landmarks()

    result = align_face(image, landmarks, output_size=224)

    assert result.aligned_image.shape == (224, 224, 3)
    assert result.output_size == 224


def test_does_not_modify_the_original_image():
    image = _random_image()
    original = image.copy()
    landmarks = _shifted_landmarks()

    align_face(image, landmarks, output_size=112)

    np.testing.assert_array_equal(image, original)


def test_aligning_twice_from_the_same_inputs_is_deterministic():
    image = _random_image()
    landmarks = _shifted_landmarks()

    first = align_face(image, landmarks, output_size=112)
    second = align_face(image, landmarks, output_size=112)

    np.testing.assert_array_equal(first.aligned_image, second.aligned_image)


@pytest.mark.parametrize(
    "bad_landmarks",
    [
        None,
        np.zeros((4, 2), dtype=np.float32),  # wrong count
        np.zeros((5, 3), dtype=np.float32),  # wrong dimensionality
        np.full((5, 2), np.nan, dtype=np.float32),  # non-finite
        np.full((5, 2), np.inf, dtype=np.float32),  # non-finite
        np.full((5, 2), 10.0, dtype=np.float32),  # degenerate: all points collapsed
    ],
)
def test_invalid_or_missing_landmarks_raise_clear_error(bad_landmarks):
    image = _random_image()

    with pytest.raises(InvalidImageError):
        align_face(image, bad_landmarks, output_size=112)


@pytest.mark.parametrize(
    "bad_image",
    [
        None,
        np.array([]),
        np.zeros((10, 10), dtype=np.uint8),  # missing channel dimension
    ],
)
def test_invalid_image_raises_clear_error(bad_image):
    with pytest.raises(InvalidImageError):
        align_face(bad_image, _shifted_landmarks(), output_size=112)


def test_non_positive_output_size_raises_clear_error():
    with pytest.raises(InvalidImageError):
        align_face(_random_image(), _shifted_landmarks(), output_size=0)
