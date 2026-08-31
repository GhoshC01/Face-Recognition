from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.exceptions import InvalidImageError
from app.utils.image_utils import crop_box

# Reason codes returned in QualityResult.reasons. Kept as plain strings (not an
# enum) so they serialize directly into API responses and log lines.
REASON_LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
REASON_INVALID_FACE_CROP = "invalid_face_crop"
REASON_FACE_TOO_SMALL = "face_too_small"
REASON_FACE_AREA_RATIO_TOO_LOW = "face_area_ratio_too_low"
REASON_IMAGE_TOO_DARK = "image_too_dark"
REASON_IMAGE_OVEREXPOSED = "image_overexposed"
REASON_IMAGE_TOO_BLURRY = "image_too_blurry"


@dataclass
class QualityThresholds:
    """Configurable pass/fail cutoffs for the quality module. Every field has a
    sane default so the checker is usable out of the box, but callers (e.g.
    app/config/settings.py) are expected to override these per deployment."""

    min_detection_confidence: float = 0.5
    min_face_width_px: int = 60
    min_face_height_px: int = 60
    min_face_area_ratio: float = 0.02
    min_brightness: float = 40.0
    max_brightness: float = 230.0
    min_sharpness: float = 60.0


# Compare is a live-capture / shared-photo path: size, brightness, and
# detection-confidence gates would reject usable frames. Only a near-zero
# sharpness floor remains (plus structural invalid_face_crop).
_COMPARE_DISABLED_FACE_PX = 1
_COMPARE_DISABLED_AREA_RATIO = 0.0
_COMPARE_DISABLED_CONFIDENCE = 0.0
_COMPARE_DISABLED_MIN_BRIGHTNESS = 0.0
_COMPARE_DISABLED_MAX_BRIGHTNESS = 255.0
DEFAULT_COMPARE_MIN_SHARPNESS = 8.0


def lenient_quality_thresholds(*, min_sharpness: float = DEFAULT_COMPARE_MIN_SHARPNESS) -> QualityThresholds:
    """Thresholds that only fail an unusable (extremely blurry) crop."""
    return QualityThresholds(
        min_detection_confidence=_COMPARE_DISABLED_CONFIDENCE,
        min_face_width_px=_COMPARE_DISABLED_FACE_PX,
        min_face_height_px=_COMPARE_DISABLED_FACE_PX,
        min_face_area_ratio=_COMPARE_DISABLED_AREA_RATIO,
        min_brightness=_COMPARE_DISABLED_MIN_BRIGHTNESS,
        max_brightness=_COMPARE_DISABLED_MAX_BRIGHTNESS,
        min_sharpness=min_sharpness,
    )


@dataclass
class QualityObservedMetrics:
    """Raw measurements the checker computed, independent of pass/fail."""

    detection_confidence: float
    face_width: int
    face_height: int
    face_area_ratio: float
    brightness: float
    sharpness: float


@dataclass
class QualityResult:
    """Structured, serializable outcome of a quality check.

    Deliberately named `quality_score`, not "accuracy" — this is a composite
    image/capture-quality signal, not a measure of recognition correctness.
    """

    accepted: bool
    quality_score: float
    reasons: list[str] = field(default_factory=list)
    metrics: QualityObservedMetrics | None = None


class QualityChecker:
    """Reusable, HRMS-independent quality gate that runs after face detection
    (SCRFD) and before embedding (MobileFaceNet). It never marks attendance
    and never talks to FAISS — it only judges whether a detected face crop is
    good enough to embed.

    All checks run to completion (no short-circuiting) so every applicable
    rejection reason is reported together, e.g. a small and blurry face
    reports both `face_too_small` and `image_too_blurry` in one result.
    """

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    @staticmethod
    def _brightness(gray: np.ndarray) -> float:
        return float(np.mean(gray))

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

    @staticmethod
    def _safe_crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
        try:
            return crop_box(image, box)
        except InvalidImageError:
            return None

    def _composite_score(self, metrics: QualityObservedMetrics) -> float:
        t = self.thresholds

        mid_brightness = (t.min_brightness + t.max_brightness) / 2
        half_range = max((t.max_brightness - t.min_brightness) / 2, 1e-6)
        brightness_score = max(0.0, 1.0 - abs(metrics.brightness - mid_brightness) / half_range)

        sharpness_score = min(1.0, metrics.sharpness / max(t.min_sharpness * 2, 1e-6))
        face_ratio_score = min(1.0, metrics.face_area_ratio / max(t.min_face_area_ratio * 3, 1e-9))
        confidence_score = min(1.0, max(0.0, metrics.detection_confidence))

        # Equal weighting across the four independent signals; each is already
        # normalized to [0, 1] so a straight average keeps the composite
        # interpretable as "average sub-score".
        composite = (brightness_score + sharpness_score + face_ratio_score + confidence_score) / 4.0
        return round(min(1.0, max(0.0, composite)), 2)

    def evaluate(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
        detection_confidence: float,
    ) -> QualityResult:
        reasons: list[str] = []
        t = self.thresholds

        if detection_confidence < t.min_detection_confidence:
            reasons.append(REASON_LOW_DETECTION_CONFIDENCE)

        crop = self._safe_crop(image, box)
        if crop is None:
            reasons.append(REASON_INVALID_FACE_CROP)
            return QualityResult(accepted=False, quality_score=0.0, reasons=reasons, metrics=None)

        face_h, face_w = crop.shape[:2]
        img_h, img_w = image.shape[:2]
        face_area_ratio = (face_w * face_h) / float(img_w * img_h) if img_w and img_h else 0.0

        if face_w < t.min_face_width_px or face_h < t.min_face_height_px:
            reasons.append(REASON_FACE_TOO_SMALL)
        if face_area_ratio < t.min_face_area_ratio:
            reasons.append(REASON_FACE_AREA_RATIO_TOO_LOW)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        brightness = self._brightness(gray)
        if brightness < t.min_brightness:
            reasons.append(REASON_IMAGE_TOO_DARK)
        elif brightness > t.max_brightness:
            reasons.append(REASON_IMAGE_OVEREXPOSED)

        sharpness = self._sharpness(gray)
        if sharpness < t.min_sharpness:
            reasons.append(REASON_IMAGE_TOO_BLURRY)

        metrics = QualityObservedMetrics(
            detection_confidence=detection_confidence,
            face_width=face_w,
            face_height=face_h,
            face_area_ratio=face_area_ratio,
            brightness=brightness,
            sharpness=sharpness,
        )

        return QualityResult(
            accepted=len(reasons) == 0,
            quality_score=self._composite_score(metrics),
            reasons=reasons,
            metrics=metrics,
        )


_REASON_FALLBACK = {
    REASON_LOW_DETECTION_CONFIDENCE: "face detection confidence is too low",
    REASON_INVALID_FACE_CROP: "detected face region is invalid",
    REASON_FACE_TOO_SMALL: "face is too small",
    REASON_FACE_AREA_RATIO_TOO_LOW: "face occupies too little of the image",
    REASON_IMAGE_TOO_DARK: "image is too dark",
    REASON_IMAGE_OVEREXPOSED: "image is overexposed",
    REASON_IMAGE_TOO_BLURRY: "image is too blurry",
}


def quality_metrics_as_dict(metrics: QualityObservedMetrics | None) -> dict[str, float | int] | None:
    """JSON-safe copy of observed metrics (native Python types only)."""
    if metrics is None:
        return None
    return {
        "detection_confidence": float(metrics.detection_confidence),
        "face_width": int(metrics.face_width),
        "face_height": int(metrics.face_height),
        "face_area_ratio": float(metrics.face_area_ratio),
        "brightness": float(metrics.brightness),
        "sharpness": float(metrics.sharpness),
    }


def describe_quality_failure(result: QualityResult, thresholds: QualityThresholds) -> str:
    """Human-readable summary of why a quality gate rejected a face crop.

    Includes observed vs minimum values so the caller can tell *how* a check
    failed (e.g. 48x52px vs a 60x60px floor), not just the reason code.
    """
    parts: list[str] = []
    m = result.metrics
    t = thresholds
    for reason in result.reasons:
        if reason == REASON_FACE_TOO_SMALL and m is not None:
            parts.append(
                f"face is too small ({m.face_width}x{m.face_height}px; "
                f"minimum {t.min_face_width_px}x{t.min_face_height_px}px)"
            )
        elif reason == REASON_FACE_AREA_RATIO_TOO_LOW and m is not None:
            parts.append(
                f"face occupies too little of the image "
                f"({m.face_area_ratio:.1%}; minimum {t.min_face_area_ratio:.1%})"
            )
        elif reason == REASON_LOW_DETECTION_CONFIDENCE and m is not None:
            parts.append(
                f"face detection confidence is too low "
                f"({m.detection_confidence:.2f}; minimum {t.min_detection_confidence:.2f})"
            )
        elif reason == REASON_IMAGE_TOO_DARK and m is not None:
            parts.append(
                f"image is too dark (brightness {m.brightness:.0f}; minimum {t.min_brightness:.0f})"
            )
        elif reason == REASON_IMAGE_OVEREXPOSED and m is not None:
            parts.append(
                f"image is overexposed (brightness {m.brightness:.0f}; maximum {t.max_brightness:.0f})"
            )
        elif reason == REASON_IMAGE_TOO_BLURRY and m is not None:
            parts.append(
                f"image is too blurry (sharpness {m.sharpness:.0f}; minimum {t.min_sharpness:.0f})"
            )
        else:
            parts.append(_REASON_FALLBACK.get(reason, reason.replace("_", " ")))
    if not parts:
        return "Face image did not pass quality checks"
    return "Face image did not pass quality checks: " + "; ".join(parts)
