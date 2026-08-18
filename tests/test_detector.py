from __future__ import annotations

import numpy as np
import pytest

from app.core.detector import FaceDetector
from app.core.exceptions import InvalidImageError

# With input_size=32, SCRFD's three FPN strides (8, 16, 32) produce feature
# grids of 4x4, 2x2, and 1x1 locations respectively, each with 2 anchors per
# location -> (32, 8, 2) total anchor slots per stride. Anchor centers follow
# FaceDetector._anchor_centers: for stride `s`, grid index k = h * width + w
# maps to raw center (x=w*s, y=h*s), and each raw center occupies two
# consecutive slots (one per anchor) in that stride's flat arrays.
_STRIDE_SLOT_COUNTS = {8: 32, 16: 8, 32: 2}


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession so detector decode logic
    (anchor decode, thresholding, NMS) can be unit tested without a real
    det_500m.onnx file."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = outputs

    def get_inputs(self):
        return [_FakeInput("input.1")]

    def run(self, output_names, feed):
        return self._outputs


def _stride8_slot(h: int, w: int, anchor: int = 0) -> int:
    k = h * 4 + w
    return k * 2 + anchor


def _build_outputs(high_score_slots_by_stride: dict[int, list[int]]) -> list[np.ndarray]:
    strides = (8, 16, 32)
    scores, bboxes, kpss = [], [], []

    for stride in strides:
        n = _STRIDE_SLOT_COUNTS[stride]
        score = np.zeros((n, 1), dtype=np.float32)
        for slot in high_score_slots_by_stride.get(stride, []):
            score[slot, 0] = 0.99
        scores.append(score)
        bboxes.append(np.zeros((n, 4), dtype=np.float32))
        kpss.append(np.zeros((n, 10), dtype=np.float32))

    return scores + bboxes + kpss


def _make_detector(outputs: list[np.ndarray]) -> FaceDetector:
    detector = FaceDetector(model_path="unused.onnx", input_size=32, confidence_threshold=0.5)
    detector._session = _FakeSession(outputs)
    detector._use_kps = True
    return detector


def _square_image(size: int = 32) -> np.ndarray:
    return np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_detects_a_single_face():
    slot = _stride8_slot(h=0, w=2)  # raw center (x=16, y=0)
    detector = _make_detector(_build_outputs({8: [slot]}))

    faces = detector.detect(_square_image())

    assert len(faces) == 1
    face = faces[0]
    assert face.score == pytest.approx(0.99, abs=1e-3)
    assert face.box == (16, 0, 16, 0)
    assert face.landmarks.shape == (5, 2)


def test_detects_multiple_faces():
    slot_a = _stride8_slot(h=0, w=2)  # (x=16, y=0)
    slot_b = _stride8_slot(h=2, w=2)  # (x=16, y=16)
    detector = _make_detector(_build_outputs({8: [slot_a, slot_b]}))

    faces = detector.detect(_square_image())

    assert len(faces) == 2
    boxes = {face.box for face in faces}
    assert boxes == {(16, 0, 16, 0), (16, 16, 16, 16)}


def test_returns_empty_list_when_no_face_present():
    detector = _make_detector(_build_outputs({}))

    faces = detector.detect(_square_image())

    assert faces == []


def test_confidence_threshold_is_configurable():
    slot = _stride8_slot(h=0, w=2)
    outputs = _build_outputs({})
    outputs[0][slot, 0] = 0.55  # below a stricter threshold, above the default

    lenient = _make_detector(outputs)
    lenient.confidence_threshold = 0.5
    assert len(lenient.detect(_square_image())) == 1

    strict = _make_detector(outputs)
    strict.confidence_threshold = 0.9
    assert strict.detect(_square_image()) == []


@pytest.mark.parametrize(
    "bad_image",
    [
        None,
        np.array([]),
        np.zeros((10, 10), dtype=np.uint8),  # missing channel dimension
        np.zeros((10, 10, 4), dtype=np.uint8),  # wrong channel count
        np.zeros((1, 1, 3), dtype=np.uint8),  # too small to process
    ],
)
def test_invalid_image_raises_clear_error(bad_image):
    detector = _make_detector(_build_outputs({}))

    with pytest.raises(InvalidImageError):
        detector.detect(bad_image)
