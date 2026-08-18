from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    external_id: str
    similarity_score: float


@dataclass
class _StoreState:
    next_id: int = 0
    # faiss_id -> external_id
    id_to_external: dict[int, str] = field(default_factory=dict)
    # external_id -> list of faiss_ids (an identity may have multiple enrolled embeddings)
    external_to_ids: dict[str, list[int]] = field(default_factory=dict)
    enrolled_at: dict[int, str] = field(default_factory=dict)


class VectorStore:
    """Reusable FAISS-backed store for L2-normalized face embeddings.

    Uses IndexIDMap2 over IndexFlatIP: since every stored and query vector is
    L2-normalized, inner product is exactly cosine similarity, and IndexIDMap2
    lets individual vectors be removed by id (plain IndexFlatIP supports
    neither stable ids nor removal on its own).

    The FAISS index (vectors) and a JSON sidecar (id <-> external_id mapping,
    enrollment timestamps) are persisted as two separate files. This class
    knows nothing about employees, HRMS, or attendance -- only opaque
    external_id -> vector(s) mappings and FAISS mechanics.
    """

    def __init__(
        self,
        dimension: int,
        index_dir: str,
        metadata_dir: str,
        index_filename: str,
        metadata_filename: str,
    ) -> None:
        self.dimension = dimension
        self.index_path = Path(index_dir) / index_filename
        self.metadata_path = Path(metadata_dir) / metadata_filename
        self._lock = threading.Lock()
        self._state = _StoreState()
        self._index = self._empty_index()
        self.load()

    def _empty_index(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def load(self) -> None:
        """Load the index and metadata from disk, if present.

        Missing files are normal (first run) and simply leave a fresh, empty
        store. A file that exists but fails to read/parse, or whose contents
        are internally inconsistent with the other file, is treated as
        corrupted: it is logged and the store falls back to a fresh empty
        state rather than operating on partial or mismatched data, or
        crashing the process that's trying to load it.
        """
        with self._lock:
            self._cleanup_stale_temp_files()
            index = self._load_index_safely()
            state = self._load_metadata_safely()

            if index.ntotal != len(state.id_to_external):
                logger.warning(
                    "vector_store_inconsistent_state_detected",
                    extra={"index_count": index.ntotal, "metadata_count": len(state.id_to_external)},
                )
                index = self._empty_index()
                state = _StoreState()

            self._index = index
            self._state = state

    def _load_index_safely(self) -> faiss.Index:
        if not self.index_path.is_file():
            return self._empty_index()

        try:
            index = faiss.read_index(str(self.index_path))
        except Exception as exc:  # faiss raises its own (non-Python) error types on bad files
            logger.warning(
                "faiss_index_load_failed", extra={"path": str(self.index_path), "reason": str(exc)}
            )
            return self._empty_index()

        if getattr(index, "d", self.dimension) != self.dimension:
            logger.warning(
                "faiss_index_dimension_mismatch",
                extra={"path": str(self.index_path), "index_dimension": index.d, "expected": self.dimension},
            )
            return self._empty_index()

        return index

    def _load_metadata_safely(self) -> _StoreState:
        if not self.metadata_path.is_file():
            return _StoreState()

        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return _StoreState(
                next_id=raw.get("next_id", 0),
                id_to_external={int(k): v for k, v in raw.get("id_to_external", {}).items()},
                external_to_ids={k: list(v) for k, v in raw.get("external_to_ids", {}).items()},
                enrolled_at={int(k): v for k, v in raw.get("enrolled_at", {}).items()},
            )
        except (json.JSONDecodeError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                "vector_store_metadata_load_failed",
                extra={"path": str(self.metadata_path), "reason": str(exc)},
            )
            return _StoreState()

    def save(self) -> None:
        """Persist the index and metadata to disk. Each file is written to a
        uniquely-named temporary file with restrictive permissions, then
        atomically renamed into place -- so a crash or kill mid-write leaves
        the previous, still-valid file on disk instead of a truncated one,
        and the embeddings/identity mappings (derived biometric data) are
        never briefly world-readable while being written.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self._atomic_write(self.index_path, prefix=".tmp_index_", writer=lambda p: faiss.write_index(self._index, p))

        payload = {
            "next_id": self._state.next_id,
            "id_to_external": self._state.id_to_external,
            "external_to_ids": self._state.external_to_ids,
            "enrolled_at": self._state.enrolled_at,
        }
        self._atomic_write(
            self.metadata_path,
            prefix=".tmp_metadata_",
            writer=lambda p: Path(p).write_text(json.dumps(payload, indent=2), encoding="utf-8"),
        )

    @staticmethod
    def _atomic_write(final_path: Path, prefix: str, writer) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=str(final_path.parent), prefix=prefix, suffix=".tmp")
        os.close(fd)
        try:
            writer(tmp_name)
            try:
                os.chmod(tmp_name, 0o600)
            except OSError:
                pass  # best-effort; not every platform/filesystem honors POSIX permission bits
            os.replace(tmp_name, final_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _cleanup_stale_temp_files(self) -> None:
        """Best-effort removal of orphaned temp files left behind by a
        process that crashed mid-save in a previous run."""
        for directory in {self.index_path.parent, self.metadata_path.parent}:
            if not directory.is_dir():
                continue
            for stray in directory.glob(".tmp_*"):
                try:
                    stray.unlink()
                except OSError:
                    pass

    def add_embedding(self, external_id: str, embedding: np.ndarray) -> int:
        if embedding.shape[-1] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[-1]} != store dimension {self.dimension}")

        with self._lock:
            faiss_id = self._state.next_id
            self._state.next_id += 1

            vector = embedding.astype(np.float32).reshape(1, -1)
            ids = np.array([faiss_id], dtype=np.int64)
            self._index.add_with_ids(vector, ids)

            self._state.id_to_external[faiss_id] = external_id
            self._state.external_to_ids.setdefault(external_id, []).append(faiss_id)
            self._state.enrolled_at[faiss_id] = datetime.now(timezone.utc).isoformat()

            self.save()

        logger.info("embedding_enrolled", extra={"external_id": external_id, "faiss_id": faiss_id})
        return faiss_id

    def has_identity(self, external_id: str) -> bool:
        with self._lock:
            return bool(self._state.external_to_ids.get(external_id))

    def get_embeddings(self, external_id: str) -> list[np.ndarray]:
        """Return every embedding enrolled under external_id (an identity may
        have more than one, e.g. multiple angles)."""
        with self._lock:
            ids = self._state.external_to_ids.get(external_id, [])
            return [self._index.reconstruct(faiss_id) for faiss_id in ids]

    def list_identities(self) -> list[str]:
        """All external_ids currently enrolled. Supports data-retention
        tooling (e.g. scripts/purge_stale_enrollments.py) without exposing
        anything beyond identity keys -- never embeddings."""
        with self._lock:
            return list(self._state.external_to_ids.keys())

    def get_last_enrolled_at(self, external_id: str) -> str | None:
        """ISO-8601 timestamp of the most recent embedding enrolled for
        external_id, or None if not enrolled. Minimizing biometric data
        retention requires knowing how old an enrollment is; this is the
        only piece of information needed for that decision, so it's exposed
        directly rather than requiring a caller to read raw metadata."""
        with self._lock:
            ids = self._state.external_to_ids.get(external_id, [])
            timestamps = [self._state.enrolled_at[i] for i in ids if i in self._state.enrolled_at]
            return max(timestamps) if timestamps else None

    def remove_embedding(self, external_id: str) -> int:
        """Remove all embeddings enrolled under external_id. Returns the
        number of embeddings removed (0 if the identity was not enrolled)."""
        ids = self._state.external_to_ids.get(external_id)
        if not ids:
            return 0

        with self._lock:
            selector = faiss.IDSelectorArray(np.array(ids, dtype=np.int64))
            self._index.remove_ids(selector)

            for faiss_id in ids:
                self._state.id_to_external.pop(faiss_id, None)
                self._state.enrolled_at.pop(faiss_id, None)
            del self._state.external_to_ids[external_id]

            self.save()

        logger.info("identity_removed", extra={"external_id": external_id, "count": len(ids)})
        return len(ids)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[MatchCandidate]:
        """Rank enrolled identities by cosine similarity to `embedding`.
        Returns an empty list when the store has nothing enrolled, or when no
        result survives to top_k -- callers apply their own similarity
        threshold on top of these raw scores to decide match/no-match."""
        with self._lock:
            if self._index.ntotal == 0:
                return []

            vector = embedding.astype(np.float32).reshape(1, -1)
            # Over-fetch since multiple faiss ids can belong to the same external_id.
            fetch_k = min(self._index.ntotal, max(top_k * 4, top_k))
            scores, ids = self._index.search(vector, fetch_k)

            best_per_identity: dict[str, float] = {}
            for score, faiss_id in zip(scores[0], ids[0]):
                if faiss_id < 0:
                    continue
                external_id = self._state.id_to_external.get(int(faiss_id))
                if external_id is None:
                    continue
                if external_id not in best_per_identity or score > best_per_identity[external_id]:
                    best_per_identity[external_id] = float(score)

        ranked = sorted(best_per_identity.items(), key=lambda pair: pair[1], reverse=True)
        return [MatchCandidate(external_id=eid, similarity_score=score) for eid, score in ranked[:top_k]]
