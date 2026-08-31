from __future__ import annotations

import numpy as np
import pytest

from app.core.quality import QualityChecker, QualityThresholds, describe_quality_failure, lenient_quality_thresholds


def make_checker(**overrides) -> QualityChecker:
    defaults = dict(
        min_detection_confidence=0.5,
        min_face_width_px=60,
        min_face_height_px=60,
        min_face_area_ratio=0.02,
        min_brightness=40.0,
        max_brightness=230.0,
        min_sharpness=60.0,
    )
    defaults.update(overrides)
    return QualityChecker(QualityThresholds(**defaults))


def well_lit_sharp_image(size: int = 200) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(80, 180, size=(size, size, 3), dtype=np.uint8)


def test_accepts_a_well_conditioned_face():
    checker = make_checker()
    image = well_lit_sharp_image(200)

    result = checker.evaluate(image, box=(0, 0, 200, 200), detection_confidence=0.95)

    assert result.accepted is True
    assert result.reasons == []
    assert 0.0 <= result.quality_score <= 1.0
    assert result.metrics is not None


def test_rejects_low_detection_confidence():
    checker = make_checker()
    image = well_lit_sharp_image(200)

    result = checker.evaluate(image, box=(0, 0, 200, 200), detection_confidence=0.2)

    assert result.accepted is False
    assert "low_detection_confidence" in result.reasons


def test_rejects_too_dark_image():
    checker = make_checker()
    image = np.full((200, 200, 3), 5, dtype=np.uint8)

    result = checker.evaluate(image, box=(0, 0, 200, 200), detection_confidence=0.9)

    assert result.accepted is False
    assert "image_too_dark" in result.reasons


def test_rejects_overexposed_image():
    checker = make_checker()
    image = np.full((200, 200, 3), 250, dtype=np.uint8)

    result = checker.evaluate(image, box=(0, 0, 200, 200), detection_confidence=0.9)

    assert result.accepted is False
    assert "image_overexposed" in result.reasons


def test_rejects_blurry_image():
    checker = make_checker()
    image = np.full((200, 200, 3), 128, dtype=np.uint8)  # flat -> zero Laplacian variance

    result = checker.evaluate(image, box=(0, 0, 200, 200), detection_confidence=0.9)

    assert result.accepted is False
    assert "image_too_blurry" in result.reasons


def test_rejects_small_face_below_absolute_pixel_minimum():
    checker = make_checker(min_face_area_ratio=0.0)  # isolate the absolute-size check
    image = well_lit_sharp_image(1000)

    result = checker.evaluate(image, box=(0, 0, 30, 30), detection_confidence=0.9)

    assert result.accepted is False
    assert "face_too_small" in result.reasons


def test_describe_quality_failure_includes_observed_face_size():
    checker = make_checker(min_face_area_ratio=0.0)
    image = well_lit_sharp_image(1000)
    result = checker.evaluate(image, box=(0, 0, 30, 30), detection_confidence=0.9)

    message = describe_quality_failure(result, checker.thresholds)

    assert "face is too small" in message
    assert "30x30px" in message
    assert "minimum 60x60px" in message


def test_rejects_small_face_relative_to_frame():
    checker = make_checker(min_face_width_px=1, min_face_height_px=1)  # isolate the ratio check
    image = well_lit_sharp_image(1000)

    result = checker.evaluate(image, box=(0, 0, 50, 50), detection_confidence=0.9)

    assert result.accepted is False
    assert "face_area_ratio_too_low" in result.reasons


def test_rejects_invalid_crop_without_crashing():
    checker = make_checker()
    image = well_lit_sharp_image(200)

    result = checker.evaluate(image, box=(50, 50, 50, 50), detection_confidence=0.9)

    assert result.accepted is False
    assert result.reasons == ["invalid_face_crop"]
    assert result.quality_score == 0.0
    assert result.metrics is None


def test_multiple_simultaneous_failures_are_all_reported():
    checker = make_checker(min_face_area_ratio=0.0)
    small_blurry_dark_image = np.full((1000, 1000, 3), 10, dtype=np.uint8)

    result = checker.evaluate(small_blurry_dark_image, box=(0, 0, 20, 20), detection_confidence=0.9)

    assert result.accepted is False
    assert set(result.reasons) == {"face_too_small", "image_too_dark", "image_too_blurry"}


def test_thresholds_are_configurable():
    lenient = make_checker(min_brightness=0.0)
    strict = make_checker(min_brightness=200.0)
    image = well_lit_sharp_image(200)

    assert "image_too_dark" not in lenient.evaluate(image, (0, 0, 200, 200), 0.9).reasons
    assert "image_too_dark" in strict.evaluate(image, (0, 0, 200, 200), 0.9).reasons


def test_result_shape_matches_contract():
    checker = make_checker()
    image = well_lit_sharp_image(200)

    result = checker.evaluate(image, box=(0, 0, 200, 200), detection_confidence=0.95)

    assert hasattr(result, "accepted")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "reasons")
    assert isinstance(result.reasons, list)


def test_lenient_thresholds_accept_small_dark_and_low_confidence_faces():
    checker = QualityChecker(lenient_quality_thresholds())
    dark_small = np.full((1000, 1000, 3), 10, dtype=np.uint8)
    dark_small[::2, ::2] = 25

    result = checker.evaluate(dark_small, box=(0, 0, 30, 30), detection_confidence=0.1)

    assert result.accepted is True
    assert result.reasons == []


def test_lenient_thresholds_reject_only_extremely_blurry_crops():
    checker = QualityChecker(lenient_quality_thresholds())
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)

    result = checker.evaluate(flat, box=(0, 0, 200, 200), detection_confidence=0.9)

    assert result.accepted is False
    assert result.reasons == ["image_too_blurry"]
