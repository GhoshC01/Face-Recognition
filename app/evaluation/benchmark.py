from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.exceptions import FaceServiceError
from app.core.recognizer import FaceRecognizer
from app.core.vector_store import VectorStore
from app.evaluation.dataset import EvaluationDataset, GalleryItem, ProbeItem, load_image
from app.evaluation.metrics import ProbeRecord, ThresholdMetrics, select_best_threshold

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    dataset_gallery_size: int
    dataset_probe_count: int
    genuine_probe_count: int
    impostor_probe_count: int
    invalid_probe_count: int
    gallery_rejected_count: int
    records: list[ProbeRecord]
    threshold_sweep: list[ThresholdMetrics]
    selected_threshold: float
    selected_metrics: ThresholdMetrics
    objective: str


class BenchmarkRunner:
    """Offline accuracy benchmark for the FaceVerification pipeline.

    Runs every probe through the exact same production pipeline
    (`FaceRecognizer`: SCRFD -> quality -> alignment -> MobileFaceNet -> L2
    normalize) that the live API uses -- this module does not reimplement
    any recognition logic, only the measurement around it.

    By default this never touches production FAISS: it builds an isolated,
    temporary gallery from the dataset's own gallery images and discards it
    afterward. Passing an already-populated `production_vector_store`
    explicitly switches to reading (searching) that store instead -- an
    opt-in for evaluating the system as currently deployed -- but even then
    only read-only methods (`search`) are ever called on it; nothing is
    enrolled or removed.
    """

    def __init__(self, recognizer: FaceRecognizer, production_vector_store: VectorStore | None = None) -> None:
        self.recognizer = recognizer
        self.production_vector_store = production_vector_store

    def run(
        self,
        dataset: EvaluationDataset,
        thresholds: list[float],
        objective: str = "accuracy",
        storage_dir: str | None = None,
    ) -> BenchmarkReport:
        gallery_rejected_count = 0
        cleanup_dir: Path | None = None

        if self.production_vector_store is not None:
            if dataset.gallery:
                logger.warning(
                    "benchmark_gallery_ignored_in_production_readonly_mode",
                    extra={"gallery_size": len(dataset.gallery)},
                )
            gallery_store = self.production_vector_store
        else:
            gallery_store, cleanup_dir, gallery_rejected_count = self._build_isolated_gallery(
                dataset.gallery, storage_dir
            )

        try:
            records = [self._evaluate_probe(probe, gallery_store) for probe in dataset.probes]
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        best_threshold, best_metrics, sweep = select_best_threshold(records, thresholds, objective=objective)

        return BenchmarkReport(
            dataset_gallery_size=len(dataset.gallery),
            dataset_probe_count=len(dataset.probes),
            genuine_probe_count=dataset.genuine_probe_count,
            impostor_probe_count=dataset.impostor_probe_count,
            invalid_probe_count=sum(1 for r in records if not r.valid),
            gallery_rejected_count=gallery_rejected_count,
            records=records,
            threshold_sweep=sweep,
            selected_threshold=best_threshold,
            selected_metrics=best_metrics,
            objective=objective,
        )

    def _build_isolated_gallery(
        self, gallery_items: list[GalleryItem], storage_dir: str | None
    ) -> tuple[VectorStore, Path | None, int]:
        processed: list[tuple[str, np.ndarray]] = []
        rejected_count = 0

        for item in gallery_items:
            try:
                image = load_image(item.image_path)
                result = self.recognizer.process(image, strict_single_face=True)
            except FaceServiceError as exc:
                rejected_count += 1
                logger.warning(
                    "benchmark_gallery_image_rejected",
                    extra={
                        "external_id": item.external_id,
                        "path": str(item.image_path),
                        "reason": exc.error_code,
                    },
                )
                continue
            processed.append((item.external_id, result.embedding))

        if not processed:
            raise ValueError("No gallery image could be processed; cannot build an evaluation gallery")

        dimension = processed[0][1].shape[0]

        if storage_dir is not None:
            index_dir = Path(storage_dir) / "faiss"
            metadata_dir = Path(storage_dir) / "metadata"
            cleanup_dir = None
        else:
            temp_root = Path(tempfile.mkdtemp(prefix="face_eval_"))
            index_dir = temp_root / "faiss"
            metadata_dir = temp_root / "metadata"
            cleanup_dir = temp_root

        gallery_store = VectorStore(
            dimension=dimension,
            index_dir=str(index_dir),
            metadata_dir=str(metadata_dir),
            index_filename="index.faiss",
            metadata_filename="metadata.json",
        )
        for external_id, embedding in processed:
            gallery_store.add_embedding(external_id, embedding)

        return gallery_store, cleanup_dir, rejected_count

    def _evaluate_probe(self, probe: ProbeItem, gallery_store: VectorStore) -> ProbeRecord:
        try:
            image = load_image(probe.image_path)
            result = self.recognizer.process(image, strict_single_face=True)
        except FaceServiceError as exc:
            return ProbeRecord(
                probe_path=str(probe.image_path),
                ground_truth_id=probe.external_id,
                predicted_id=None,
                similarity=None,
                valid=False,
                rejection_reason=exc.error_code,
            )

        candidates = gallery_store.search(result.embedding, top_k=1)
        best = candidates[0] if candidates else None

        return ProbeRecord(
            probe_path=str(probe.image_path),
            ground_truth_id=probe.external_id,
            predicted_id=best.external_id if best else None,
            similarity=best.similarity_score if best else None,
            valid=True,
        )
