from __future__ import annotations

import numpy as np
import pytest

from app.core.embedding import FaceEmbedder, _infer_embedding_dimension
from app.core.exceptions import InvalidImageError, ModelNotReadyError


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOutput:
    def __init__(self, shape) -> None:
        self.shape = shape


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession so embed() logic (blob
    preprocessing, dimension enforcement, normalization) can be unit tested
    without a real w600k_mbf.onnx file."""

    def __init__(self, output_shape=(1, 512), raw_output: np.ndarray | None = None) -> None:
        self._output_shape = output_shape
        dimension = output_shape[-1]
        self._raw_output = raw_output if raw_output is not None else np.ones((1, dimension), dtype=np.float32)
        self.last_feed: dict | None = None

    def get_inputs(self):
        return [_FakeInput("input.1")]

    def get_outputs(self):
        return [_FakeOutput(self._output_shape)]

    def run(self, output_names, feed):
        self.last_feed = feed
        return [self._raw_output]


def _make_embedder(output_shape=(1, 512), raw_output: np.ndarray | None = None, input_size: int = 112) -> tuple[FaceEmbedder, _FakeSession]:
    embedder = FaceEmbedder(model_path="unused.onnx", input_size=input_size)
    fake_session = _FakeSession(output_shape=output_shape, raw_output=raw_output)
    embedder._session = fake_session
    embedder._input_name = "input.1"
    embedder._embedding_dimension = _infer_embedding_dimension(output_shape)
    return embedder, fake_session


def _aligned_face(size: int = 112) -> np.ndarray:
    return np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)


# --- dimension discovery: never assumed, always read from the model ---


@pytest.mark.parametrize(
    "output_shape,expected",
    [
        ((1, 512), 512),
        (("batch", 512), 512),
        ((512,), 512),
        ((1, 128), 128),
    ],
)
def test_infers_embedding_dimension_from_model_output_shape(output_shape, expected):
    assert _infer_embedding_dimension(output_shape) == expected


@pytest.mark.parametrize("bad_shape", [(), (1, "unknown"), (1, 0), (1, -1), (1, None)])
def test_rejects_non_fixed_or_invalid_output_shape(bad_shape):
    with pytest.raises(ModelNotReadyError):
        _infer_embedding_dimension(bad_shape)


# --- embed(): preprocessing, output contract, dimension enforcement ---


def test_embed_returns_l2_normalized_vector_of_model_declared_length():
    raw = np.random.default_rng(1).normal(size=(1, 512)).astype(np.float32)
    embedder, _ = _make_embedder(output_shape=(1, 512), raw_output=raw)

    embedding = embedder.embed(_aligned_face())

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_embed_respects_a_different_model_dimension():
    raw = np.random.default_rng(2).normal(size=(1, 128)).astype(np.float32)
    embedder, _ = _make_embedder(output_shape=(1, 128), raw_output=raw)

    embedding = embedder.embed(_aligned_face())

    assert embedding.shape == (128,)
    assert embedder.embedding_dimension == 128


def test_embed_sends_correctly_shaped_preprocessed_blob():
    embedder, fake_session = _make_embedder(input_size=112)

    embedder.embed(_aligned_face(size=200))  # input image size != model input size

    blob = fake_session.last_feed["input.1"]
    assert blob.shape == (1, 3, 112, 112)  # NCHW, resized to the configured input size
    assert blob.dtype == np.float32


def test_embed_raises_when_model_output_does_not_match_declared_dimension():
    # Simulate a model whose declared output shape says 512 but whose actual
    # run-time output is a different length -- a real-world drift/mismatch case.
    mismatched_output = np.ones((1, 256), dtype=np.float32)
    embedder, _ = _make_embedder(output_shape=(1, 512), raw_output=mismatched_output)

    with pytest.raises(ModelNotReadyError):
        embedder.embed(_aligned_face())


@pytest.mark.parametrize(
    "bad_face",
    [
        None,
        np.array([]),
        np.zeros((112, 112), dtype=np.uint8),  # missing channel dimension
        np.zeros((112, 112, 4), dtype=np.uint8),  # wrong channel count
    ],
)
def test_embed_rejects_invalid_aligned_face(bad_face):
    embedder, _ = _make_embedder()

    with pytest.raises(InvalidImageError):
        embedder.embed(bad_face)


def test_embed_does_not_reload_an_already_loaded_model():
    embedder, _ = _make_embedder()
    load_calls = []
    embedder.load = lambda: load_calls.append(1)  # would only run if _session were None

    embedder.embed(_aligned_face())
    embedder.embed(_aligned_face())

    assert load_calls == []


def test_embedding_dimension_is_none_before_load():
    embedder = FaceEmbedder(model_path="unused.onnx")
    assert embedder.embedding_dimension is None
    assert embedder.is_loaded is False


def test_load_raises_file_not_found_for_missing_model(tmp_path):
    embedder = FaceEmbedder(model_path=str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError):
        embedder.load()


def test_embed_wraps_missing_model_as_model_not_ready(tmp_path):
    embedder = FaceEmbedder(model_path=str(tmp_path / "missing.onnx"))
    with pytest.raises(ModelNotReadyError):
        embedder.embed(_aligned_face())
