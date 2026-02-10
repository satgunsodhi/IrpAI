"""
YOLO-based UI element detector.

Uses Ultralytics YOLO26 (or YOLOv8) to detect UI elements like buttons,
text fields, icons, headers, navigation bars, etc.

For best results, fine-tune on a UI-element dataset such as:
    - Rico (Android UI)
    - WebUI
    - Custom screenshot annotations

Out of the box this uses the general COCO-trained model and maps
common COCO classes to approximate UI roles.  A fine-tuned model is
strongly recommended for production use.

Install
-------
    pip install ultralytics
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from irpai.config import DetectionConfig
from irpai.detection.base import BaseDetector, DetectionOutput, DetectionResult
from irpai.utils.image import pil_to_cv2
from irpai.utils.logging import get_logger

log = get_logger("detection.yolo")

# Rough mapping of COCO classes → UI-relevant categories.
# A fine-tuned model on UI data would not need this.
_COCO_TO_UI: dict[str, str] = {
    "person": "avatar",
    "cell phone": "device_frame",
    "laptop": "device_frame",
    "tv": "screen_area",
    "monitor": "screen_area",
    "book": "content_area",
    "clock": "widget",
    "keyboard": "input_area",
    "mouse": "input_area",
}


class YOLODetector(BaseDetector):
    """
    YOLO-based UI element detector implementing :class:`BaseDetector`.

    Lazily loads the model on first call to :meth:`detect`.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.cfg = config or DetectionConfig()
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"YOLO ({self.cfg.model_name})"

    # -- lazy init ----------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            ) from exc

        weights = self.cfg.weights_path
        if weights is not None:
            model_path = str(weights)
            log.info("Loading YOLO weights from %s", model_path)
        else:
            # Map friendly names to ultralytics model IDs
            model_map: dict[str, str] = {
                "yolo26n": "yolo11n.pt",
                "yolo26s": "yolo11s.pt",
                "yolo26m": "yolo11m.pt",
                "yolo26l": "yolo11l.pt",
                "yolo26x": "yolo11x.pt",
                "yolov8n": "yolov8n.pt",
                "yolov8s": "yolov8s.pt",
            }
            model_path = model_map.get(self.cfg.model_name, f"{self.cfg.model_name}.pt")
            log.info("Loading YOLO model: %s (device=%s)", model_path, self.cfg.device)

        self._model = YOLO(model_path)
        return self._model

    def warmup(self) -> None:
        """Pre-load the model and run a dummy inference to warm caches."""
        model = self._ensure_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, device=self.cfg.device, verbose=False)
        log.info("YOLO warmup complete")

    # -- detection ----------------------------------------------------------

    def detect(self, image: Image.Image | np.ndarray) -> DetectionOutput:
        model = self._ensure_model()

        if isinstance(image, Image.Image):
            w, h = image.size
            arr = pil_to_cv2(image)
        else:
            h, w = image.shape[:2]
            arr = image

        t0 = time.perf_counter()

        results = model.predict(
            arr,
            device=self.cfg.device,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections: list[DetectionResult] = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Map class name
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                    ui_label = _COCO_TO_UI.get(class_name, class_name)

                    bbox = (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                    )

                    detections.append(DetectionResult(
                        label=ui_label,
                        bbox=bbox,
                        confidence=conf,
                        meta={"coco_class": class_name, "class_id": cls_id},
                    ))

        output = DetectionOutput(
            detections=detections,
            image_size=(w, h),
            model_name=self.name,
            inference_ms=elapsed_ms,
        )

        log.info(
            "YOLO: %d detections in %.1f ms",
            len(detections),
            elapsed_ms,
        )
        return output


class FallbackGridDetector(BaseDetector):
    """
    Smart rule-based fallback detector for the *lite* tier.

    Instead of blindly dividing the image into a fixed grid, this detector
    finds actual UI boundaries by looking for:

    1. **Horizontal colour bands** — title bars, status bars, nav bars
       (full-width rows with a uniform colour distinct from neighbours).
    2. **Vertical colour boundaries** — sidebars / panels
       (full-height columns with a uniform colour change).
    3. Any remaining large area is treated as the main content region.

    No ML model required — works on pixel statistics alone.
    """

    def __init__(self, rows: int = 3, cols: int = 3) -> None:
        # rows/cols kept for API compat but not used by the smart path
        self.rows = rows
        self.cols = cols

    @property
    def name(self) -> str:
        return "SmartFallbackDetector"

    def detect(self, image: Image.Image | np.ndarray) -> DetectionOutput:
        if isinstance(image, np.ndarray):
            arr = image
            h, w = arr.shape[:2]
        else:
            arr = np.array(image)
            w, h = image.size

        detections = self._find_ui_regions(arr, w, h)

        return DetectionOutput(
            detections=detections,
            image_size=(w, h),
            model_name=self.name,
            inference_ms=0.0,
        )

    # ------------------------------------------------------------------

    def _find_ui_regions(
        self, arr: np.ndarray, w: int, h: int,
    ) -> list[DetectionResult]:
        """Detect UI regions via colour-band analysis."""
        detections: list[DetectionResult] = []

        # --- Horizontal bands (title bar, status bar, tabs) ---------------
        h_splits = self._find_horizontal_splits(arr, w, h)

        # --- Vertical splits (sidebar) ------------------------------------
        v_splits = self._find_vertical_splits(arr, w, h)

        # Build regions from the split grid
        y_edges = sorted(set([0] + h_splits + [h]))
        x_edges = sorted(set([0] + v_splits + [w]))

        for yi in range(len(y_edges) - 1):
            for xi in range(len(x_edges) - 1):
                x1, x2 = x_edges[xi], x_edges[xi + 1]
                y1, y2 = y_edges[yi], y_edges[yi + 1]

                # Skip tiny slivers
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                label = self._classify_region(x1, y1, x2, y2, w, h)

                detections.append(DetectionResult(
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=1.0,
                    meta={"source": "color_split"},
                ))

        # If nothing was detected, return a single full-image region
        if not detections:
            detections.append(DetectionResult(
                label="content_area",
                bbox=(0, 0, w, h),
                confidence=1.0,
                meta={"source": "full_image"},
            ))

        return detections

    # ------------------------------------------------------------------

    def _find_horizontal_splits(
        self, arr: np.ndarray, w: int, h: int,
    ) -> list[int]:
        """Find y-positions where the full-width colour changes sharply.

        Uses a sliding-window approach: compare each row's mean colour to
        the mean of the preceding *band* of rows.  This avoids false splits
        from single-pixel aliasing/text edges.
        """
        band = 4  # rows to average for stability
        row_means = arr.mean(axis=1).astype(float)  # (h, 3)

        splits: list[int] = []
        prev_band = row_means[:band].mean(axis=0)

        for y in range(band, h - band):
            curr_band = row_means[y : y + band].mean(axis=0)
            diff = float(np.linalg.norm(curr_band - prev_band))
            if diff > 25:
                if not splits or (y - splits[-1]) > 20:
                    splits.append(y)
                    # Jump ahead past the transition
                    prev_band = row_means[y + band : y + 2 * band].mean(axis=0) if y + 2 * band < h else curr_band
                    continue
            prev_band = curr_band

        return splits

    def _find_vertical_splits(
        self, arr: np.ndarray, w: int, h: int,
    ) -> list[int]:
        """Find x-positions where the full-height colour changes sharply.

        Same band-averaging approach as horizontal, but checks that the
        colour change is consistent across most of the image height (not
        just a local text edge).
        """
        band = 4
        col_means = arr.mean(axis=0).astype(float)  # (w, 3)

        splits: list[int] = []
        prev_band = col_means[:band].mean(axis=0)

        for x in range(band, w - band):
            curr_band = col_means[x : x + band].mean(axis=0)
            diff = float(np.linalg.norm(curr_band - prev_band))

            if diff > 10:
                # Verify this is a *structural* boundary, not just a text edge:
                # check that the colour change is consistent in ≥ 60 % of rows
                left_strip = arr[:, max(0, x - 2) : x, :].mean(axis=1).astype(float)
                right_strip = arr[:, x : min(w, x + 2), :].mean(axis=1).astype(float)
                per_row_diff = np.linalg.norm(right_strip - left_strip, axis=1)
                consistent = float((per_row_diff > 8).mean())

                if consistent > 0.55 and (not splits or (x - splits[-1]) > 20):
                    splits.append(x)
                    prev_band = col_means[x + band : x + 2 * band].mean(axis=0) if x + 2 * band < w else curr_band
                    continue
            prev_band = curr_band

        return splits

    @staticmethod
    def _classify_region(
        x1: int, y1: int, x2: int, y2: int, w: int, h: int,
    ) -> str:
        """Heuristic label based on position and aspect ratio."""
        rw = x2 - x1
        rh = y2 - y1
        aspect = rw / max(rh, 1)

        # Top full-width thin strip -> title / tab bar
        if y1 < h * 0.1 and rw > w * 0.8 and aspect > 4:
            return "header"

        # Bottom full-width thin strip -> status bar
        if y2 > h * 0.9 and rw > w * 0.8 and aspect > 4:
            return "status_bar"

        # Tall narrow left strip -> sidebar
        if x1 < w * 0.05 and rh > h * 0.5 and rw < w * 0.35:
            return "sidebar"

        # Tall narrow right strip -> side panel
        if x2 > w * 0.95 and rh > h * 0.5 and rw < w * 0.35:
            return "sidebar"

        # Otherwise -> content area
        return "content_area"
