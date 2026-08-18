from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.utils.image_utils import decode_image_bytes


@dataclass
class GalleryItem:
    """A reference/enrollment-style image used to build the evaluation
    gallery -- distinct from, and never written into, production FAISS."""

    external_id: str
    image_path: Path


@dataclass
class ProbeItem:
    """A labeled test/query image. `external_id` is the ground truth
    identity, or None if this probe is an impostor -- a person who is not
    present in the gallery at all. Impostor probes are required to measure
    False Accept Rate correctly; without them FAR cannot be computed."""

    external_id: str | None
    image_path: Path


@dataclass
class EvaluationDataset:
    gallery: list[GalleryItem]
    probes: list[ProbeItem]

    @property
    def genuine_probe_count(self) -> int:
        return sum(1 for p in self.probes if p.external_id is not None)

    @property
    def impostor_probe_count(self) -> int:
        return sum(1 for p in self.probes if p.external_id is None)


def load_dataset_from_manifest(manifest_path: str | Path) -> EvaluationDataset:
    """Loads a JSON manifest describing an evaluation dataset:

    {
      "gallery": [
        {"external_id": "EMP001", "image_path": "gallery/emp001.jpg"}
      ],
      "probes": [
        {"external_id": "EMP001", "image_path": "probes/emp001_test1.jpg"},
        {"external_id": null, "image_path": "probes/stranger1.jpg"}
      ]
    }

    A null/omitted `external_id` on a probe marks it as an impostor (not
    enrolled anywhere) rather than a genuine test of a real identity.
    Relative `image_path` values are resolved against the manifest file's
    own directory, so a dataset folder can be moved around as a unit.
    """
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    def _resolve(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else (base_dir / path)

    gallery = [
        GalleryItem(external_id=item["external_id"], image_path=_resolve(item["image_path"]))
        for item in raw.get("gallery", [])
    ]
    probes = [
        ProbeItem(external_id=item.get("external_id"), image_path=_resolve(item["image_path"]))
        for item in raw.get("probes", [])
    ]

    if not gallery:
        raise ValueError(f"Dataset manifest '{manifest_path}' has no gallery entries")
    if not probes:
        raise ValueError(f"Dataset manifest '{manifest_path}' has no probe entries")

    return EvaluationDataset(gallery=gallery, probes=probes)


def load_image(path: str | Path) -> np.ndarray:
    return decode_image_bytes(Path(path).read_bytes())
