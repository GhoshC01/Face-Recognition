from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import (
    IdentityNotFoundError,
    InvalidFrameCountError,
    LowImageQualityError,
    NoFaceDetectedError,
)
from app.core.quality import QualityObservedMetrics, QualityResult
from app.core.recognizer import FaceEmbeddingResult
from app.core.vector_store import VectorStore
from app.services.verification_service import VerificationService
from tests.conftest import make_synthetic_image

DIMENSION = 8


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=DIMENSION).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _nudged(base: np.ndarray, seed: int, noise_scale: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nudged = base + rng.normal(scale=noise_scale, size=base.shape).astype(np.float32)
    return nudged / np.linalg.norm(nudged)


def _result(embedding: np.ndarray, detection_score: float = 0.95) -> FaceEmbeddingResult:
    return FaceEmbeddingResult(
        embedding=embedding,
        detection_score=detection_score,
        box=(0, 0, 100, 100),
        quality=QualityResult(
            accepted=True,
            quality_score=0.9,
            reasons=[],
            metrics=QualityObservedMetrics(
                detection_confidence=detection_score,
                face_width=100,
                face_height=100,
                face_area_ratio=0.5,
                brightness=120.0,
                sharpness=150.0,
            ),
        ),
    )


class ScriptedRecognizer:
    """Maps a marker byte embedded in a synthetic image to a scripted
    outcome -- a FaceEmbeddingResult to return, or an exception to raise --
    so per-frame branching in verify_multi_frame can be tested
    deterministically without real ONNX models."""

    def __init__(self, outcomes: dict[int, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[int] = []

    def process(self, image: np.ndarray, strict_single_face: bool = False):
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        outcome = self.outcomes[marker]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(DIMENSION, str(tmp_path / "faiss"), str(tmp_path / "metadata"), "index.faiss", "metadata.json")


def _service(recognizer, store, **overrides) -> VerificationService:
    defaults = dict(
        verification_threshold=0.80,
        identification_threshold=0.80,
        identification_top_k=5,
        multi_frame_min_frames=3,
        multi_frame_max_frames=5,
        multi_frame_min_valid_frames=2,
        multi_frame_min_agreeing_frames=2,
        multi_frame_consensus_ratio=0.6,
    )
    defaults.update(overrides)
    return VerificationService(recognizer, store, **defaults)


def _frames(*markers: int) -> list[np.ndarray]:
    return [make_synthetic_image(m) for m in markers]


# --- the example from the prompt: 3 consistent, high-similarity frames -> PASS ---


def test_three_consistent_frames_pass(store):
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    recognizer = ScriptedRecognizer(
        {1: _result(_nudged(base, 11)), 2: _result(_nudged(base, 12)), 3: _result(_nudged(base, 13))}
    )
    service = _service(recognizer, store)

    response = service.verify_multi_frame(_frames(1, 2, 3), external_id="EMP001")

    assert response.status == "PASS"
    assert response.verified is True
    assert response.external_id == "EMP001"
    assert response.frames_submitted == 3
    assert response.frames_valid == 3
    assert response.frames_agreeing == 3
    assert response.frames is None  # debug not requested


# --- invalid/low-quality frames are ignored, not fatal ---


def test_invalid_frames_are_ignored_not_fatal(store):
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    recognizer = ScriptedRecognizer(
        {
            1: _result(_nudged(base, 11)),
            2: NoFaceDetectedError("no face"),
            3: _result(_nudged(base, 13)),
            4: LowImageQualityError("too blurry"),
            5: _result(_nudged(base, 15)),
        }
    )
    service = _service(recognizer, store)

    response = service.verify_multi_frame(_frames(1, 2, 3, 4, 5), external_id="EMP001")

    assert response.status == "PASS"
    assert response.frames_submitted == 5
    assert response.frames_valid == 3  # frames 2 and 4 ignored
    assert response.frames_agreeing == 3


def test_decode_failures_are_ignored_as_invalid_frames(store):
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    recognizer = ScriptedRecognizer({1: _result(_nudged(base, 11)), 3: _result(_nudged(base, 13))})
    service = _service(recognizer, store)

    images = [make_synthetic_image(1), None, make_synthetic_image(3)]  # None == failed to decode
    response = service.verify_multi_frame(images, external_id="EMP001")

    assert response.status == "PASS"
    assert response.frames_valid == 2
    assert response.frames_agreeing == 2


# --- consensus ratio and the "never pass on one weak frame" floor ---


def test_fails_when_consensus_ratio_not_met(store):
    """Isolates the ratio gate from the absolute min_agreeing_frames floor:
    2 agreeing frames clears the default floor (2) but not the default
    consensus ratio (0.6) once split across 4 valid frames (2/4 = 0.5)."""
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    stranger = _unit_vector(99)  # dissimilar -> won't clear threshold against EMP001
    recognizer = ScriptedRecognizer(
        {
            1: _result(_nudged(base, 11)),
            2: _result(_nudged(base, 12)),
            3: _result(stranger),
            4: _result(stranger),
        }
    )
    service = _service(recognizer, store)  # default consensus_ratio=0.6, min_agreeing_frames=2

    response = service.verify_multi_frame(_frames(1, 2, 3, 4), external_id="EMP001")

    assert response.status == "FAIL"
    assert response.frames_valid == 4
    assert response.frames_agreeing == 2
    assert "consensus_ratio_not_met" in response.reasons


def test_single_agreeing_frame_never_passes_even_if_ratio_would_allow_it(store):
    """A ratio-only check could be fooled by a single valid frame achieving
    100% agreement; multi_frame_min_agreeing_frames is the absolute floor
    that stops one strong (or lucky) frame from carrying a PASS alone."""
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    recognizer = ScriptedRecognizer({1: _result(_nudged(base, 11)), 2: NoFaceDetectedError("no face")})
    service = _service(
        recognizer,
        store,
        multi_frame_min_frames=2,
        multi_frame_min_valid_frames=1,  # deliberately lenient, to isolate the agreeing-frames floor
        multi_frame_consensus_ratio=0.5,
        multi_frame_min_agreeing_frames=2,
    )

    response = service.verify_multi_frame(_frames(1, 2), external_id="EMP001")

    assert response.status == "FAIL"
    assert response.frames_valid == 1
    assert response.frames_agreeing == 1
    assert "insufficient_agreeing_frames" in response.reasons


def test_fails_when_too_few_valid_frames(store):
    store.add_embedding("EMP001", _unit_vector(1))
    recognizer = ScriptedRecognizer(
        {1: NoFaceDetectedError("no face"), 2: NoFaceDetectedError("no face"), 3: NoFaceDetectedError("no face")}
    )
    service = _service(recognizer, store)

    response = service.verify_multi_frame(_frames(1, 2, 3), external_id="EMP001")

    assert response.status == "FAIL"
    assert response.frames_valid == 0
    assert response.external_id is None
    assert "insufficient_valid_frames" in response.reasons


# --- Mode B vs Mode A external_id echoing ---


def test_mode_b_echoes_external_id_even_on_fail(store):
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    stranger = _unit_vector(99)
    recognizer = ScriptedRecognizer({1: _result(stranger), 2: _result(stranger), 3: _result(stranger)})
    service = _service(recognizer, store)

    response = service.verify_multi_frame(_frames(1, 2, 3), external_id="EMP001")

    assert response.status == "FAIL"
    assert response.external_id == "EMP001"


def test_mode_a_identification_majority_vote(store):
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    other = _unit_vector(50)
    store.add_embedding("EMP002", other)

    recognizer = ScriptedRecognizer(
        {1: _result(_nudged(base, 11)), 2: _result(_nudged(base, 12)), 3: _result(_nudged(other, 13))}
    )
    service = _service(recognizer, store)

    response = service.verify_multi_frame(_frames(1, 2, 3))  # no external_id -> Mode A

    assert response.status == "PASS"
    assert response.mode == "identification"
    assert response.external_id == "EMP001"  # 2 of 3 frames agree on EMP001


def test_mode_a_fail_returns_null_external_id(store):
    stranger = _unit_vector(99)
    recognizer = ScriptedRecognizer({1: _result(stranger), 2: _result(stranger), 3: _result(stranger)})
    service = _service(recognizer, store)  # nothing enrolled at all

    response = service.verify_multi_frame(_frames(1, 2, 3))

    assert response.status == "FAIL"
    assert response.external_id is None
    assert "no_matching_identity" in response.reasons


# --- frame count validation ---


def test_rejects_too_few_frames(store):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1))})
    service = _service(recognizer, store)

    with pytest.raises(InvalidFrameCountError):
        service.verify_multi_frame(_frames(1), external_id="EMP001")


def test_rejects_too_many_frames(store):
    recognizer = ScriptedRecognizer({i: _result(_unit_vector(i)) for i in range(1, 7)})
    service = _service(recognizer, store)

    with pytest.raises(InvalidFrameCountError):
        service.verify_multi_frame(_frames(1, 2, 3, 4, 5, 6), external_id="EMP001")


# --- unknown identity fails fast, before touching any frame ---


def test_unknown_external_id_fails_before_processing_frames(store):
    recognizer = ScriptedRecognizer({1: _result(_unit_vector(1)), 2: _result(_unit_vector(2)), 3: _result(_unit_vector(3))})
    service = _service(recognizer, store)

    with pytest.raises(IdentityNotFoundError):
        service.verify_multi_frame(_frames(1, 2, 3), external_id="ghost")

    assert recognizer.calls == []


# --- debug mode ---


def test_debug_mode_includes_frame_diagnostics(store):
    base = _unit_vector(1)
    store.add_embedding("EMP001", base)
    recognizer = ScriptedRecognizer(
        {1: _result(_nudged(base, 11)), 2: NoFaceDetectedError("no face"), 3: _result(_nudged(base, 13))}
    )
    service = _service(recognizer, store)

    response = service.verify_multi_frame(_frames(1, 2, 3), external_id="EMP001", debug=True)

    assert response.frames is not None
    assert len(response.frames) == 3
    assert response.frames[1].valid is False
    assert response.frames[1].rejection_reason == "no_face_detected"
    assert response.frames[0].valid is True
    assert response.frames[0].passed_threshold is True
