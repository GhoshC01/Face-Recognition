from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.core.exceptions import NoFaceDetectedError
from app.core.quality import QualityObservedMetrics, QualityResult
from app.core.recognizer import FaceEmbeddingResult
from app.core.vector_store import VectorStore
from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.dataset import load_dataset_from_manifest

DIMENSION = 8


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=DIMENSION).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _nudged(base: np.ndarray, seed: int, noise_scale: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nudged = base + rng.normal(scale=noise_scale, size=base.shape).astype(np.float32)
    return nudged / np.linalg.norm(nudged)


def _result(embedding: np.ndarray) -> FaceEmbeddingResult:
    return FaceEmbeddingResult(
        embedding=embedding,
        detection_score=0.95,
        box=(0, 0, 100, 100),
        quality=QualityResult(
            accepted=True,
            quality_score=0.9,
            reasons=[],
            metrics=QualityObservedMetrics(
                detection_confidence=0.95, face_width=100, face_height=100,
                face_area_ratio=0.5, brightness=120.0, sharpness=150.0,
            ),
        ),
    )


class ScriptedRecognizer:
    """Maps a marker byte embedded in a real on-disk image file to a
    scripted outcome, so the benchmark runner's file-loading and metrics
    wiring can be exercised end-to-end without real ONNX models."""

    def __init__(self, outcomes: dict[int, object]) -> None:
        self.outcomes = outcomes

    def process(self, image: np.ndarray, strict_single_face: bool = False):
        marker = int(image[0, 0, 0])
        outcome = self.outcomes[marker]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _write_marker_image(path, marker: int) -> None:
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    image[0, 0, 0] = marker
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())


def _write_dataset(tmp_path, gallery: list[tuple[str, int]], probes: list[tuple[str | None, int]]):
    (tmp_path / "gallery").mkdir()
    (tmp_path / "probes").mkdir()

    manifest = {"gallery": [], "probes": []}
    for external_id, marker in gallery:
        filename = f"gallery/{external_id}_{marker}.png"
        _write_marker_image(tmp_path / filename, marker)
        manifest["gallery"].append({"external_id": external_id, "image_path": filename})

    for external_id, marker in probes:
        filename = f"probes/probe_{marker}.png"
        _write_marker_image(tmp_path / filename, marker)
        manifest["probes"].append({"external_id": external_id, "image_path": filename})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_dataset_from_manifest(str(manifest_path))


def test_isolated_gallery_full_run(tmp_path):
    base_a = _unit_vector(1)
    base_b = _unit_vector(2)
    stranger = _unit_vector(99)

    dataset = _write_dataset(
        tmp_path,
        gallery=[("EMP001", 1), ("EMP002", 2)],
        probes=[
            ("EMP001", 11),  # genuine, should match EMP001
            ("EMP002", 12),  # genuine, should match EMP002
            (None, 13),  # impostor
        ],
    )

    recognizer = ScriptedRecognizer(
        {
            1: _result(base_a),
            2: _result(base_b),
            11: _result(_nudged(base_a, 11)),
            12: _result(_nudged(base_b, 12)),
            13: _result(stranger),
        }
    )
    runner = BenchmarkRunner(recognizer=recognizer)

    report = runner.run(dataset, thresholds=[0.3, 0.5, 0.7], objective="accuracy")

    assert report.dataset_gallery_size == 2
    assert report.dataset_probe_count == 3
    assert report.genuine_probe_count == 2
    assert report.impostor_probe_count == 1
    assert report.invalid_probe_count == 0
    assert report.gallery_rejected_count == 0
    assert len(report.threshold_sweep) == 3
    assert report.selected_metrics.accuracy == max(m.accuracy for m in report.threshold_sweep)
    # at a reasonable threshold both genuine probes should match their own identity
    best_at_0_5 = next(m for m in report.threshold_sweep if m.threshold == 0.5)
    assert best_at_0_5.correct >= 2


def test_probe_pipeline_failure_is_recorded_as_invalid(tmp_path):
    base_a = _unit_vector(1)
    dataset = _write_dataset(
        tmp_path,
        gallery=[("EMP001", 1)],
        probes=[("EMP001", 11), ("EMP001", 12)],
    )
    recognizer = ScriptedRecognizer(
        {1: _result(base_a), 11: _result(_nudged(base_a, 11)), 12: NoFaceDetectedError("no face")}
    )
    runner = BenchmarkRunner(recognizer=recognizer)

    report = runner.run(dataset, thresholds=[0.5])

    assert report.invalid_probe_count == 1
    invalid_record = next(r for r in report.records if not r.valid)
    assert invalid_record.rejection_reason == "no_face_detected"
    assert invalid_record.similarity is None


def test_gallery_image_rejected_is_excluded_not_fatal(tmp_path):
    base_a = _unit_vector(1)
    dataset = _write_dataset(
        tmp_path,
        gallery=[("EMP001", 1), ("EMP002", 2)],
        probes=[("EMP001", 11)],
    )
    recognizer = ScriptedRecognizer(
        {1: _result(base_a), 2: NoFaceDetectedError("no face"), 11: _result(_nudged(base_a, 11))}
    )
    runner = BenchmarkRunner(recognizer=recognizer)

    report = runner.run(dataset, thresholds=[0.5])

    assert report.gallery_rejected_count == 1  # EMP002's gallery image was unusable, but the run still completed
    assert report.dataset_gallery_size == 2  # reflects what was submitted, not what was usable


def test_all_gallery_images_rejected_raises(tmp_path):
    dataset = _write_dataset(tmp_path, gallery=[("EMP001", 1)], probes=[("EMP001", 11)])
    recognizer = ScriptedRecognizer({1: NoFaceDetectedError("no face"), 11: _result(_unit_vector(1))})
    runner = BenchmarkRunner(recognizer=recognizer)

    with pytest.raises(ValueError):
        runner.run(dataset, thresholds=[0.5])


def test_production_readonly_mode_never_writes_to_the_store(tmp_path):
    base_a = _unit_vector(1)
    production_store = VectorStore(
        DIMENSION, str(tmp_path / "prod_faiss"), str(tmp_path / "prod_metadata"), "index.faiss", "metadata.json"
    )
    production_store.add_embedding("EMP001", base_a)
    count_before = production_store.count()

    # dataset still declares a gallery -- it must be ignored in this mode.
    dataset = _write_dataset(tmp_path, gallery=[("EMP999", 99)], probes=[("EMP001", 11)])
    recognizer = ScriptedRecognizer({99: _result(_unit_vector(50)), 11: _result(_nudged(base_a, 11))})
    runner = BenchmarkRunner(recognizer=recognizer, production_vector_store=production_store)

    report = runner.run(dataset, thresholds=[0.5])

    assert production_store.count() == count_before  # never modified
    assert not production_store.has_identity("EMP999")  # dataset gallery was ignored
    assert report.records[0].predicted_id == "EMP001"  # search still worked against production data
