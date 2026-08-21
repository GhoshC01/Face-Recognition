from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.exceptions import InvalidImageError, ModelNotReadyError
from app.utils.timing import Stopwatch

logger = logging.getLogger(__name__)


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in original image coordinates
    score: float
    landmarks: np.ndarray  # shape (5, 2): left-eye, right-eye, nose, left-mouth, right-mouth

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= thresh)[0]
        order = order[remaining + 1]

    return keep


class FaceDetector:
    """ONNX Runtime wrapper for an SCRFD-family face detector (e.g. det_500m.onnx).

    Decodes the standard SCRFD output layout: for each of the three FPN strides
    (8, 16, 32) the model emits a classification score tensor, a box-distance
    tensor, and (if the export includes keypoints) a landmark-distance tensor,
    in that fixed order. This matches the models shipped in InsightFace's
    buffalo_sc / antelope model packs.
    """

    _feat_stride_fpn = (8, 16, 32)
    _num_anchors = 2

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ) -> None:
        self.model_path = model_path
        self.input_size = (input_size, input_size)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._session: ort.InferenceSession | None = None
        self._use_kps = True
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return

        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Detector model not found at '{path}'")

        sw = Stopwatch()
        options = ort.SessionOptions()
        if self._intra_op_threads:
            options.intra_op_num_threads = self._intra_op_threads
        if self._inter_op_threads:
            options.inter_op_num_threads = self._inter_op_threads

        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        num_outputs = len(self._session.get_outputs())
        self._use_kps = num_outputs == 9
        logger.info(
            "detector_loaded",
            extra={"model_path": str(path), "num_outputs": num_outputs, "load_ms": sw.lap_ms()},
        )

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        img_h, img_w = image.shape[:2]
        im_ratio = img_h / img_w
        model_w, model_h = self.input_size
        model_ratio = model_h / model_w

        if im_ratio > model_ratio:
            new_h = model_h
            new_w = int(round(new_h / im_ratio))
        else:
            new_w = model_w
            new_h = int(round(new_w * im_ratio))

        det_scale = new_h / img_h
        resized = cv2.resize(image, (new_w, new_h))

        padded = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        return blob, det_scale

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached

        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        centers = np.stack([centers] * self._num_anchors, axis=1).reshape((-1, 2))

        if len(self._center_cache) < 100:
            self._center_cache[key] = centers
        return centers

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Defensive input check so a bad array fails with a clear, typed error
        instead of an opaque OpenCV/numpy exception. This is independent of
        (and in addition to) any bytes-level decoding validation callers do
        upstream — FaceDetector is meant to be usable standalone.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError("Detector received an empty or invalid image")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImageError(
                f"Detector requires an HxWx3 BGR image, got shape {getattr(image, 'shape', None)}"
            )
        if image.shape[0] < 2 or image.shape[1] < 2:
            raise InvalidImageError(f"Image is too small to process, got shape {image.shape}")

    def detect(self, image: np.ndarray) -> list[Face]:
        self._validate_image(image)

        if self._session is None:
            logger.warning("detector_lazy_load_triggered_by_request")
            try:
                self.load()
            except FileNotFoundError as exc:
                raise ModelNotReadyError("Face detector model is not available") from exc
        assert self._session is not None

        blob, det_scale = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        fmc = len(self._feat_stride_fpn)
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        input_w, input_h = self.input_size
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + fmc].reshape(-1, 4) * stride

            height, width = input_h // stride, input_w // stride
            anchor_centers = self._anchor_centers(height, width, stride)

            pos_inds = np.where(scores >= self.confidence_threshold)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if self._use_kps:
                kps_preds = outputs[idx + fmc * 2].reshape(-1, 10) * stride
                kpss = _distance2kps(anchor_centers, kps_preds).reshape(-1, 5, 2)
                kpss_list.append(kpss[pos_inds])

        if not scores_list or all(s.size == 0 for s in scores_list):
            return []

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list) / det_scale
        kpss = np.concatenate(kpss_list) / det_scale if self._use_kps else None

        order = scores.argsort()[::-1]
        dets = np.hstack([bboxes, scores[:, None]]).astype(np.float32)[order]
        if kpss is not None:
            kpss = kpss[order]

        keep = _nms(dets, self.nms_threshold)
        dets = dets[keep]
        if kpss is not None:
            kpss = kpss[keep]

        faces: list[Face] = []
        for i in range(dets.shape[0]):
            x1, y1, x2, y2, score = dets[i]
            box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            landmarks = kpss[i] if kpss is not None else np.zeros((5, 2), dtype=np.float32)
            faces.append(Face(box=box, score=float(score), landmarks=landmarks))

        return faces
