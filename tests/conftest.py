from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.core.quality import QualityObservedMetrics, QualityResult
from app.core.recognizer import FaceEmbeddingResult
from app.main import create_app


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("METADATA_DIR", str(tmp_path / "metadata"))
    monkeypatch.setenv("DETECTOR_MODEL_PATH", str(tmp_path / "missing_det.onnx"))
    monkeypatch.setenv("RECOGNIZER_MODEL_PATH", str(tmp_path / "missing_rec.onnx"))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class FakeRecognizer:
    """Deterministic stand-in for FaceRecognizer: maps a marker byte embedded in
    a synthetic image's corner pixel to a fixed embedding, so service-layer
    tests don't require real ONNX models."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def process(self, image: np.ndarray, strict_single_face: bool = False) -> FaceEmbeddingResult:
        marker = int(image[0, 0, 0])
        rng = np.random.default_rng(marker)
        embedding = rng.normal(size=self.dimension).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        return FaceEmbeddingResult(
            embedding=embedding,
            detection_score=0.99,
            box=(0, 0, 10, 10),
            quality=QualityResult(
                accepted=True,
                quality_score=0.95,
                reasons=[],
                metrics=QualityObservedMetrics(
                    detection_confidence=0.99,
                    face_width=10,
                    face_height=10,
                    face_area_ratio=0.5,
                    brightness=120.0,
                    sharpness=200.0,
                ),
            ),
        )


def make_synthetic_image(marker: int) -> np.ndarray:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    image[0, 0, 0] = marker
    return image
