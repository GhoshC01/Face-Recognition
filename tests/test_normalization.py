from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import InvalidEmbeddingError
from app.core.normalization import l2_normalize


def test_returns_unit_norm_vector():
    embedding = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5

    normalized = l2_normalize(embedding)

    assert np.linalg.norm(normalized) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(normalized, [0.6, 0.8], atol=1e-6)


@pytest.mark.parametrize("dimension", [2, 128, 256, 512])
def test_preserves_embedding_dimension(dimension):
    rng = np.random.default_rng(dimension)
    embedding = rng.normal(size=dimension).astype(np.float32)

    normalized = l2_normalize(embedding)

    assert normalized.shape == (dimension,)


def test_output_dtype_is_float32():
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    normalized = l2_normalize(embedding)

    assert normalized.dtype == np.float32


def test_does_not_mutate_the_input_array():
    embedding = np.array([3.0, 4.0], dtype=np.float32)
    original = embedding.copy()

    l2_normalize(embedding)

    np.testing.assert_array_equal(embedding, original)


def test_rejects_nan_values():
    embedding = np.array([1.0, np.nan, 3.0], dtype=np.float32)

    with pytest.raises(InvalidEmbeddingError):
        l2_normalize(embedding)


def test_rejects_infinite_values():
    embedding = np.array([1.0, np.inf, 3.0], dtype=np.float32)

    with pytest.raises(InvalidEmbeddingError):
        l2_normalize(embedding)


def test_rejects_negative_infinity():
    embedding = np.array([1.0, -np.inf, 3.0], dtype=np.float32)

    with pytest.raises(InvalidEmbeddingError):
        l2_normalize(embedding)


def test_rejects_near_zero_vector():
    embedding = np.zeros(512, dtype=np.float32)

    with pytest.raises(InvalidEmbeddingError):
        l2_normalize(embedding)


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        [1.0, 2.0, 3.0],  # plain list, not ndarray
        np.array([]),  # empty
        np.zeros((4, 4), dtype=np.float32),  # 2-D, not a vector
    ],
)
def test_rejects_invalid_shapes_and_types(bad_input):
    with pytest.raises(InvalidEmbeddingError):
        l2_normalize(bad_input)


def test_normalized_dot_product_equals_cosine_similarity():
    """The whole point of L2-normalizing before storage: dot product of two
    normalized vectors must equal the cosine similarity of the originals,
    which is exactly what an inner-product index computes natively."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=512).astype(np.float32) * 3.7  # arbitrary magnitude
    b = rng.normal(size=512).astype(np.float32) * 0.2  # different arbitrary magnitude

    cosine_similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    normalized_dot_product = float(np.dot(l2_normalize(a), l2_normalize(b)))

    assert normalized_dot_product == pytest.approx(cosine_similarity, abs=1e-5)


def test_accepts_a_custom_min_norm_threshold():
    tiny_embedding = np.full(8, 1e-8, dtype=np.float32)  # norm ~ 2.8e-8

    with pytest.raises(InvalidEmbeddingError):
        l2_normalize(tiny_embedding, min_norm=1e-6)

    # With a lenient enough floor, the same vector normalizes successfully.
    normalized = l2_normalize(tiny_embedding, min_norm=1e-9)
    assert np.linalg.norm(normalized) == pytest.approx(1.0, abs=1e-4)
