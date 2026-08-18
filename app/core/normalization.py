from __future__ import annotations

import numpy as np

from app.core.exceptions import InvalidEmbeddingError

_DEFAULT_MIN_NORM = 1e-12


def l2_normalize(embedding: np.ndarray, min_norm: float = _DEFAULT_MIN_NORM) -> np.ndarray:
    """Rescale a raw embedding vector to unit L2 norm.

    This is a small, generic numeric transform: it takes one vector and
    returns one vector of the same length. It has no knowledge of FAISS, face
    detection, or HRMS, and is reusable anywhere a raw model output needs to
    be turned into a comparison-ready embedding.

    Cosine similarity between two vectors a, b is (a . b) / (|a| |b|). Once
    every stored and query vector is pre-normalized to |v| = 1, that formula
    reduces to a plain dot product -- which is exactly what an inner-product
    index (e.g. FAISS IndexFlatIP) computes natively. Normalizing once here,
    rather than at every comparison, means:
      - similarity search becomes a single fast inner-product op with no
        per-comparison division, which is what makes IndexFlatIP correct and
        efficient for cosine similarity in the first place;
      - all stored embeddings sit on the same unit hypersphere, so distance
        reflects only the *direction* (facial identity) the model encoded,
        not the incidental *magnitude* of its raw output (which can vary
        with lighting/pose and carries no identity signal); and
      - similarity scores land in a fixed, predictable [-1, 1] range, so a
        single configured threshold behaves consistently across every
        enrolled identity instead of drifting with unnormalized magnitudes.
    """
    if embedding is None or not isinstance(embedding, np.ndarray):
        raise InvalidEmbeddingError(f"Embedding must be a numpy array, got {type(embedding)!r}")
    if embedding.ndim != 1 or embedding.size == 0:
        raise InvalidEmbeddingError(f"Embedding must be a non-empty 1-D vector, got shape {embedding.shape}")

    if not np.all(np.isfinite(embedding)):
        raise InvalidEmbeddingError(
            "Embedding contains non-finite values",
            nan_count=int(np.isnan(embedding).sum()),
            inf_count=int(np.isinf(embedding).sum()),
        )

    vector = embedding.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vector))

    if norm < min_norm:
        raise InvalidEmbeddingError("Embedding has near-zero magnitude and cannot be normalized")

    return vector / norm
