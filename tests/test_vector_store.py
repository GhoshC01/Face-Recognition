from __future__ import annotations

import numpy as np
import pytest

from app.core.vector_store import VectorStore


def _unit_vector(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dimension).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _orthogonal_pair(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors with ~zero cosine similarity, for low-similarity tests."""
    a = np.zeros(dimension, dtype=np.float32)
    b = np.zeros(dimension, dtype=np.float32)
    a[0] = 1.0
    b[1] = 1.0
    return a, b


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(
        dimension=8,
        index_dir=str(tmp_path / "faiss"),
        metadata_dir=str(tmp_path / "metadata"),
        index_filename="index.faiss",
        metadata_filename="metadata.json",
    )


# --- add / get / count ---


def test_add_embedding_and_get_embeddings(store):
    vector = _unit_vector(8, seed=1)
    faiss_id = store.add_embedding("emp-1", vector)

    assert isinstance(faiss_id, int)
    assert store.has_identity("emp-1")
    stored = store.get_embeddings("emp-1")
    assert len(stored) == 1
    np.testing.assert_allclose(stored[0], vector, atol=1e-5)


def test_count_reflects_number_of_stored_embeddings(store):
    assert store.count() == 0

    store.add_embedding("emp-1", _unit_vector(8, seed=1))
    assert store.count() == 1

    store.add_embedding("emp-2", _unit_vector(8, seed=2))
    assert store.count() == 2


def test_rejects_wrong_dimension(store):
    with pytest.raises(ValueError):
        store.add_embedding("emp-1", np.zeros(4, dtype=np.float32))


# --- multiple embeddings for one external ID ---


def test_multiple_embeddings_for_the_same_external_id(store):
    vec_a = _unit_vector(8, seed=10)
    vec_b = _unit_vector(8, seed=11)

    id_a = store.add_embedding("emp-1", vec_a)
    id_b = store.add_embedding("emp-1", vec_b)

    assert id_a != id_b
    assert store.count() == 2

    stored = store.get_embeddings("emp-1")
    assert len(stored) == 2
    stored_set = {tuple(np.round(v, 4)) for v in stored}
    assert tuple(np.round(vec_a, 4)) in stored_set
    assert tuple(np.round(vec_b, 4)) in stored_set


def test_search_returns_best_match_across_multiple_embeddings_for_one_identity(store):
    vec_a = _unit_vector(8, seed=10)
    vec_b = _unit_vector(8, seed=11)
    store.add_embedding("emp-1", vec_a)
    store.add_embedding("emp-1", vec_b)

    # Querying with vec_a exactly should surface emp-1 once (best of its two
    # embeddings), not twice.
    results = store.search(vec_a, top_k=5)

    assert len(results) == 1
    assert results[0].external_id == "emp-1"
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)


def test_removing_identity_removes_all_its_embeddings(store):
    store.add_embedding("emp-1", _unit_vector(8, seed=10))
    store.add_embedding("emp-1", _unit_vector(8, seed=11))
    store.add_embedding("emp-2", _unit_vector(8, seed=12))

    removed = store.remove_embedding("emp-1")

    assert removed == 2
    assert not store.has_identity("emp-1")
    assert store.count() == 1


# --- search ---


def test_search_returns_best_match(store):
    vec_a = _unit_vector(8, seed=1)
    vec_b = _unit_vector(8, seed=2)
    store.add_embedding("emp-a", vec_a)
    store.add_embedding("emp-b", vec_b)

    results = store.search(vec_a, top_k=2)
    assert results[0].external_id == "emp-a"
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)


def test_search_on_empty_store_returns_no_results(store):
    """An empty store (nothing enrolled yet) is the simplest 'unknown' case."""
    results = store.search(_unit_vector(8, seed=1), top_k=5)
    assert results == []


def test_search_reports_low_similarity_for_dissimilar_query(store):
    enrolled, dissimilar_query = _orthogonal_pair(8)
    store.add_embedding("emp-1", enrolled)

    results = store.search(dissimilar_query, top_k=5)

    assert len(results) == 1
    assert results[0].external_id == "emp-1"
    # Orthogonal unit vectors have cosine similarity ~0 -- a caller's
    # threshold (not the store itself) is what turns this into "unknown".
    assert results[0].similarity_score == pytest.approx(0.0, abs=1e-4)


# --- remove ---


def test_remove_embedding(store):
    vector = _unit_vector(8, seed=3)
    store.add_embedding("emp-1", vector)
    removed = store.remove_embedding("emp-1")

    assert removed == 1
    assert not store.has_identity("emp-1")
    assert store.search(vector, top_k=1) == []


def test_remove_embedding_for_unknown_identity_is_a_no_op(store):
    assert store.remove_embedding("ghost") == 0


# --- save / load ---


def test_persistence_round_trip(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"
    vector = _unit_vector(8, seed=4)

    original = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")
    original.add_embedding("emp-1", vector)

    reloaded = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")
    assert reloaded.has_identity("emp-1")
    assert reloaded.count() == 1
    np.testing.assert_allclose(reloaded.get_embeddings("emp-1")[0], vector, atol=1e-5)


def test_explicit_save_and_load_round_trip(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"
    store = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")
    store.add_embedding("emp-1", _unit_vector(8, seed=5))

    store.save()  # already saved by add_embedding, but exercise the explicit API too
    store.load()

    assert store.has_identity("emp-1")
    assert store.count() == 1


# --- missing / corrupted files ---


def test_missing_index_and_metadata_files_start_empty(tmp_path):
    store = VectorStore(8, str(tmp_path / "faiss"), str(tmp_path / "metadata"), "index.faiss", "metadata.json")
    assert store.count() == 0


def test_corrupted_index_file_is_handled_safely(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"
    faiss_dir.mkdir(parents=True)
    (faiss_dir / "index.faiss").write_bytes(b"this is not a valid faiss index")

    store = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")

    assert store.count() == 0
    # The store must still be usable after recovering from corruption.
    store.add_embedding("emp-1", _unit_vector(8, seed=6))
    assert store.count() == 1


def test_corrupted_metadata_file_is_handled_safely(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "metadata.json").write_text("{not valid json", encoding="utf-8")

    store = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")

    assert store.count() == 0
    store.add_embedding("emp-1", _unit_vector(8, seed=7))
    assert store.count() == 1


def test_index_and_metadata_disagreeing_on_count_resets_safely(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"

    # Build a valid store with two embeddings, then delete only the metadata
    # sidecar so the index and metadata fall out of sync with each other.
    store = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")
    store.add_embedding("emp-1", _unit_vector(8, seed=8))
    store.add_embedding("emp-2", _unit_vector(8, seed=9))
    (metadata_dir / "metadata.json").unlink()

    reloaded = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")

    assert reloaded.count() == 0  # inconsistent state -> safe reset, not a crash or silent corruption


def test_mismatched_index_dimension_is_handled_safely(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"

    wrong_dimension_store = VectorStore(4, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")
    wrong_dimension_store.add_embedding("emp-1", _unit_vector(4, seed=1))

    store = VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")

    assert store.count() == 0


# --- safe temp-file handling ---


def test_save_leaves_no_stray_temp_files(store, tmp_path):
    store.add_embedding("emp-1", _unit_vector(8, seed=1))
    store.save()

    leftovers = list((tmp_path / "faiss").glob(".tmp_*")) + list((tmp_path / "metadata").glob(".tmp_*"))
    assert leftovers == []


def test_orphaned_temp_files_are_cleaned_up_on_load(tmp_path):
    faiss_dir, metadata_dir = tmp_path / "faiss", tmp_path / "metadata"
    faiss_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    # Simulate leftovers from a process that crashed mid-save previously.
    (faiss_dir / ".tmp_index_abc123.tmp").write_bytes(b"partial")
    (metadata_dir / ".tmp_metadata_xyz789.tmp").write_text("{partial", encoding="utf-8")

    VectorStore(8, str(faiss_dir), str(metadata_dir), "index.faiss", "metadata.json")

    assert list(faiss_dir.glob(".tmp_*")) == []
    assert list(metadata_dir.glob(".tmp_*")) == []


# --- data retention support ---


def test_list_identities(store):
    assert store.list_identities() == []

    store.add_embedding("emp-1", _unit_vector(8, seed=1))
    store.add_embedding("emp-2", _unit_vector(8, seed=2))

    assert sorted(store.list_identities()) == ["emp-1", "emp-2"]


def test_get_last_enrolled_at_returns_none_for_unknown_identity(store):
    assert store.get_last_enrolled_at("ghost") is None


def test_get_last_enrolled_at_returns_most_recent_timestamp(store):
    store.add_embedding("emp-1", _unit_vector(8, seed=1))
    store.add_embedding("emp-1", _unit_vector(8, seed=2))  # a later, second embedding

    last_enrolled = store.get_last_enrolled_at("emp-1")

    assert last_enrolled is not None
    # Every timestamp recorded for this identity's embeddings is <= the reported "last" one.
    ids = store._state.external_to_ids["emp-1"]
    all_timestamps = [store._state.enrolled_at[i] for i in ids]
    assert last_enrolled == max(all_timestamps)
