"""Hybrid detector that combines RT-DETR with rule-based fallback logic."""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from irpai.detection.base import BaseDetector, DetectionOutput, DetectionResult
from irpai.detection.rule_fallback import FallbackGridDetector
from irpai.utils.logging import get_logger

log = get_logger("detection.hybrid")

STRUCTURAL_LABELS = {"header", "sidebar", "status_bar", "content_area"}


class HybridDetector(BaseDetector):
    """
    Two-stage detector:

    1. RT-DETR (ML) when available.
    2. Rule-based fallback for graceful degradation and structure hints.
    """

    def __init__(
        self,
        use_ml: bool = True,
        ml_model_id: str | None = None,
        confidence_threshold: float = 0.35,
        structural_boost: bool = True,
        nms_iou_threshold: float = 0.35,
        max_detections: int = 120,
        device: str = "cpu",
    ) -> None:
        self.use_ml = use_ml
        self.structural_boost = structural_boost
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

        self._rule_detector = FallbackGridDetector(max_detections=max_detections)
        self._ml_detector: BaseDetector | None = None

        if use_ml:
            try:
                from irpai.detection.rt_detr_detector import RTDETRDetector

                self._ml_detector = RTDETRDetector(
                    model_id=ml_model_id,
                    confidence_threshold=confidence_threshold,
                    device=device,
                )
            except Exception as exc:
                log.warning("Failed to initialize RT-DETR detector: %s", exc)
                self._ml_detector = None

    @property
    def name(self) -> str:
        if self._ml_detector:
            return f"HybridDetector(ml={self._ml_detector.name})"
        return "HybridDetector(ml=disabled)"

    def warmup(self) -> None:
        if self._ml_detector:
            self._ml_detector.warmup()

    def detect(self, image: Image.Image | np.ndarray) -> DetectionOutput:
        t0 = time.perf_counter()

        ml_output: DetectionOutput | None = None
        all_detections: list[DetectionResult] = []

        if self._ml_detector:
            try:
                ml_output = self._ml_detector.detect(image)
                all_detections.extend(ml_output.detections)
            except Exception:
                log.warning("ML detector failed, switching to fallback path", exc_info=True)
                ml_output = None

        need_rule = self.structural_boost or ml_output is None
        if need_rule:
            rule_output = self._rule_detector.detect(image)

            if ml_output is None:
                all_detections = list(rule_output.detections)
            else:
                for det in rule_output.detections:
                    source = str(det.meta.get("source", ""))
                    if det.label in STRUCTURAL_LABELS and source == "color_split":
                        all_detections.append(det)

        if all_detections:
            all_detections = _nms(all_detections, self.nms_iou_threshold)

        w, h = _image_size(image)
        if not all_detections:
            all_detections = [
                DetectionResult(
                    label="content_area",
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    meta={"source": "full_image_fallback"},
                )
            ]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return DetectionOutput(
            detections=all_detections[: self.max_detections],
            image_size=(w, h),
            model_name=self.name,
            inference_ms=round(elapsed_ms, 1),
        )


def _image_size(image: Image.Image | np.ndarray) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        return int(image.shape[1]), int(image.shape[0])
    return image.size


def _nms(detections: list[DetectionResult], threshold: float) -> list[DetectionResult]:
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: list[DetectionResult] = []

    for det in sorted_dets:
        if all(_compute_iou(det.bbox, selected.bbox) <= threshold for selected in keep):
            keep.append(det)

    return keep


def _compute_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
