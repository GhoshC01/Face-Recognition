from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.core.exceptions import InvalidImageError
from app.evaluation.dataset import load_dataset_from_manifest, load_image


def _write_manifest(tmp_path, manifest: dict) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def test_loads_gallery_and_probes_with_relative_paths_resolved(tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "gallery": [{"external_id": "EMP001", "image_path": "gallery/emp001.jpg"}],
            "probes": [
                {"external_id": "EMP001", "image_path": "probes/emp001_test1.jpg"},
                {"external_id": None, "image_path": "probes/stranger.jpg"},
            ],
        },
    )

    dataset = load_dataset_from_manifest(manifest_path)

    assert len(dataset.gallery) == 1
    assert dataset.gallery[0].external_id == "EMP001"
    assert dataset.gallery[0].image_path == tmp_path / "gallery" / "emp001.jpg"

    assert len(dataset.probes) == 2
    assert dataset.probes[0].external_id == "EMP001"
    assert dataset.probes[1].external_id is None
    assert dataset.probes[1].image_path == tmp_path / "probes" / "stranger.jpg"


def test_genuine_and_impostor_counts(tmp_path):
    manifest_path = _write_manifest(
        tmp_path,
        {
            "gallery": [{"external_id": "EMP001", "image_path": "g.jpg"}],
            "probes": [
                {"external_id": "EMP001", "image_path": "p1.jpg"},
                {"external_id": "EMP001", "image_path": "p2.jpg"},
                {"external_id": None, "image_path": "p3.jpg"},
            ],
        },
    )

    dataset = load_dataset_from_manifest(manifest_path)

    assert dataset.genuine_probe_count == 2
    assert dataset.impostor_probe_count == 1


def test_rejects_manifest_with_no_gallery(tmp_path):
    manifest_path = _write_manifest(tmp_path, {"gallery": [], "probes": [{"external_id": "A", "image_path": "p.jpg"}]})

    with pytest.raises(ValueError):
        load_dataset_from_manifest(manifest_path)


def test_rejects_manifest_with_no_probes(tmp_path):
    manifest_path = _write_manifest(tmp_path, {"gallery": [{"external_id": "A", "image_path": "g.jpg"}], "probes": []})

    with pytest.raises(ValueError):
        load_dataset_from_manifest(manifest_path)


def test_load_image_decodes_a_real_file(tmp_path):
    image_path = tmp_path / "sample.png"
    array = np.full((20, 20, 3), 100, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", array)
    assert ok
    image_path.write_bytes(buffer.tobytes())

    loaded = load_image(image_path)

    assert loaded.shape == (20, 20, 3)


def test_load_image_rejects_corrupted_file(tmp_path):
    image_path = tmp_path / "bad.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(InvalidImageError):
        load_image(image_path)
