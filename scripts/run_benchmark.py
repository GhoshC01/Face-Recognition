"""Offline accuracy benchmark for the FaceVerification pipeline.

Runs a labeled, independent test dataset (NOT production enrollment data)
through the exact same detector/quality/alignment/embedding pipeline the
live API uses, sweeps a set of candidate similarity thresholds, and reports
accuracy/precision/recall/FAR/FRR/confusion-matrix at each -- selecting the
best threshold from measured results rather than a guessed value.

By default this never touches production FAISS: it builds an isolated,
temporary gallery from the dataset's own gallery images. Pass
--use-production to instead read (never write) the already-enrolled
production index as the gallery.

Usage:
    python scripts/run_benchmark.py --dataset path/to/manifest.json
    python scripts/run_benchmark.py --dataset manifest.json --thresholds 0.3,0.4,0.5,0.6
    python scripts/run_benchmark.py --dataset manifest.json --objective f1 --output report.json
    python scripts/run_benchmark.py --dataset manifest.json --use-production

Dataset manifest format:
    {
      "gallery": [{"external_id": "EMP001", "image_path": "gallery/emp001.jpg"}],
      "probes": [
        {"external_id": "EMP001", "image_path": "probes/emp001_test1.jpg"},
        {"external_id": null, "image_path": "probes/stranger1.jpg"}
      ]
    }

A probe's `external_id` is the ground truth; null marks an impostor (a
person not present in the gallery at all), required to measure FAR.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.core.detector import FaceDetector  # noqa: E402
from app.core.embedding import FaceEmbedder  # noqa: E402
from app.core.quality import QualityChecker, QualityThresholds  # noqa: E402
from app.core.recognizer import FaceRecognizer  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.evaluation.benchmark import BenchmarkRunner  # noqa: E402
from app.evaluation.dataset import load_dataset_from_manifest  # noqa: E402


def _default_thresholds() -> list[float]:
    return [round(0.20 + 0.05 * i, 2) for i in range(16)]  # 0.20 .. 0.95


def _build_recognizer() -> FaceRecognizer:
    """Constructs the exact same pipeline app/main.py wires up for the live
    API, from the same settings -- so benchmark results reflect what the
    deployed service actually does, not a reimplementation of it."""
    settings = get_settings()

    detector = FaceDetector(
        model_path=settings.detector_model_path,
        input_size=settings.detector_input_size,
        confidence_threshold=settings.detector_confidence_threshold,
        nms_threshold=settings.detector_nms_threshold,
        intra_op_threads=settings.onnx_intra_op_threads,
        inter_op_threads=settings.onnx_inter_op_threads,
    )
    embedder = FaceEmbedder(
        model_path=settings.recognizer_model_path,
        input_size=settings.recognizer_input_size,
        intra_op_threads=settings.onnx_intra_op_threads,
        inter_op_threads=settings.onnx_inter_op_threads,
    )
    detector.load()
    embedder.load()

    quality_checker = QualityChecker(
        QualityThresholds(
            min_detection_confidence=settings.quality_min_detection_confidence,
            min_face_width_px=settings.quality_min_face_width_px,
            min_face_height_px=settings.quality_min_face_height_px,
            min_face_area_ratio=settings.quality_min_face_area_ratio,
            min_brightness=settings.quality_min_brightness,
            max_brightness=settings.quality_max_brightness,
            min_sharpness=settings.quality_min_sharpness,
        )
    )
    return FaceRecognizer(
        detector=detector,
        embedder=embedder,
        quality_checker=quality_checker,
        max_faces_to_consider=settings.max_faces_to_consider,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to a dataset manifest JSON file")
    parser.add_argument("--thresholds", default=None, help="Comma-separated similarity thresholds to evaluate")
    parser.add_argument("--objective", default="accuracy", choices=["accuracy", "f1", "min_far_frr_gap"])
    parser.add_argument("--output", default=None, help="Write the full JSON report to this path (default: stdout)")
    parser.add_argument(
        "--use-production",
        action="store_true",
        help="Read the production FAISS index as the gallery instead of building an isolated one "
        "from the dataset's own gallery images. Read-only: never enrolls or writes anything.",
    )
    args = parser.parse_args()

    recognizer = _build_recognizer()

    production_store = None
    if args.use_production:
        settings = get_settings()
        production_store = VectorStore(
            dimension=settings.embedding_dimension,
            index_dir=settings.faiss_index_dir,
            metadata_dir=settings.metadata_dir,
            index_filename=settings.faiss_index_filename,
            metadata_filename=settings.metadata_filename,
        )

    dataset = load_dataset_from_manifest(args.dataset)
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else _default_thresholds()

    runner = BenchmarkRunner(recognizer=recognizer, production_vector_store=production_store)
    report = runner.run(dataset, thresholds=thresholds, objective=args.objective)

    output = {
        "total_probes": report.dataset_probe_count,
        "genuine_probes": report.genuine_probe_count,
        "impostor_probes": report.impostor_probe_count,
        "invalid_probes": report.invalid_probe_count,
        "gallery_size": report.dataset_gallery_size,
        "gallery_rejected": report.gallery_rejected_count,
        "objective": report.objective,
        "selected_threshold": report.selected_threshold,
        "selected_metrics": asdict(report.selected_metrics),
        "threshold_sweep": [asdict(m) for m in report.threshold_sweep],
    }
    output_json = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Full report written to {args.output}")
    else:
        print(output_json)

    m = report.selected_metrics
    print(
        f"\nAt threshold={report.selected_threshold} (selected by '{args.objective}'):\n"
        f"  Total test images: {m.total}\n"
        f"  Correct: {m.correct}\n"
        f"  Incorrect: {m.incorrect}\n"
        f"  Rejected/Unknown: {m.rejected}\n"
        f"  Accuracy: {m.accuracy:.2%}\n"
        f"  Precision: {m.precision:.2%}  Recall: {m.recall:.2%}\n"
        f"  FAR: {m.far:.2%}  FRR: {m.frr:.2%}  Substitution errors: {m.substitution_error_rate:.2%}"
    )


if __name__ == "__main__":
    main()
