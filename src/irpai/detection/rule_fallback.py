"""
Smart rule-based UI region detector.

Combines two detection strategies for maximum coverage:
1. **Colour-band analysis** — finds structural UI boundaries (title bars,
   sidebars, status bars, content areas) via pixel-row/column statistics.
2. **Contour analysis** — finds rectangular UI elements (buttons, inputs,
   menus, modals, icons) via adaptive thresholding + contour detection.
3. **Text-region clustering** — groups connected text components into
   semantic text blocks via connected-components analysis.

All techniques use only ``numpy`` + ``opencv`` — no ML model required.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from irpai.detection.base import BaseDetector, DetectionOutput, DetectionResult
from irpai.utils.logging import get_logger

log = get_logger("detection")


class FallbackGridDetector(BaseDetector):
    """
    Smart rule-based UI detector combining colour-band analysis,
    contour-based element detection, and text-region clustering.

    Produces two categories of detections:
    - **Structural regions** (header, sidebar, status_bar, content_area)
      from colour-band analysis → confidence 1.0.
    - **UI elements** (button, text_input, text_block, menu, modal,
      icon, container) from contour analysis → confidence 0.5–0.6.

    No ML model required — works on pixel statistics alone.
    """

    def __init__(
        self,
        nms_threshold: float = 0.3,
        max_detections: int = 100,
    ) -> None:
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    @property
    def name(self) -> str:
        return "SmartFallbackDetector"

    def detect(self, image: Image.Image | np.ndarray) -> DetectionOutput:
        t0 = time.perf_counter()

        if isinstance(image, np.ndarray):
            arr = image
            h, w = arr.shape[:2]
        else:
            arr = np.array(image)
            w, h = image.size

        # --- Strategy 1: colour-band structural regions -------------------
        structural = self._find_ui_regions(arr, w, h)

        # --- Strategy 2: contour-based element detection ------------------
        # Convert to BGR for OpenCV if image is RGB (PIL gives RGB)
        if arr.ndim == 3 and arr.shape[2] == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            bgr = arr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

        contour_dets = self._detect_rectangles(gray, h, w)
        text_dets = self._detect_text_regions(gray, h, w)

        # --- Merge + NMS --------------------------------------------------
        all_dets = structural + contour_dets + text_dets
        all_dets = self._nms(all_dets, self.nms_threshold)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not all_dets:
            all_dets.append(
                DetectionResult(
                    label="content_area",
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    meta={"source": "full_image"},
                )
            )

        log.info(
            "Detection: %d regions in %.1f ms (structural=%d, contour=%d, text=%d)",
            len(all_dets),
            elapsed_ms,
            len(structural),
            len(contour_dets),
            len(text_dets),
        )

        return DetectionOutput(
            detections=all_dets[: self.max_detections],
            image_size=(w, h),
            model_name=self.name,
            inference_ms=round(elapsed_ms, 1),
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Strategy 1: colour-band structural regions
    # ══════════════════════════════════════════════════════════════════════

    def _find_ui_regions(
        self,
        arr: np.ndarray,
        w: int,
        h: int,
    ) -> list[DetectionResult]:
        """Detect structural UI regions via colour-band analysis."""
        detections: list[DetectionResult] = []

        h_splits = self._find_horizontal_splits(arr, w, h)
        v_splits = self._find_vertical_splits(arr, w, h)

        y_edges = sorted(set([0] + h_splits + [h]))
        x_edges = sorted(set([0] + v_splits + [w]))

        for yi in range(len(y_edges) - 1):
            for xi in range(len(x_edges) - 1):
                x1, x2 = x_edges[xi], x_edges[xi + 1]
                y1, y2 = y_edges[yi], y_edges[yi + 1]

                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                label = self._classify_structural_region(x1, y1, x2, y2, w, h)
                detections.append(
                    DetectionResult(
                        label=label,
                        bbox=(x1, y1, x2, y2),
                        confidence=1.0,
                        meta={"source": "color_split"},
                    )
                )

        return detections

    def _find_horizontal_splits(
        self,
        arr: np.ndarray,
        w: int,
        h: int,
    ) -> list[int]:
        """Find y-positions where the full-width colour changes sharply."""
        band = 4
        row_means = arr.mean(axis=1).astype(float)

        splits: list[int] = []
        prev_band = row_means[:band].mean(axis=0)

        for y in range(band, h - band):
            curr_band = row_means[y : y + band].mean(axis=0)
            diff = float(np.linalg.norm(curr_band - prev_band))
            if diff > 25:
                if not splits or (y - splits[-1]) > 20:
                    splits.append(y)
                    prev_band = (
                        row_means[y + band : y + 2 * band].mean(axis=0)
                        if y + 2 * band < h
                        else curr_band
                    )
                    continue
            prev_band = curr_band

        return splits

    def _find_vertical_splits(
        self,
        arr: np.ndarray,
        w: int,
        h: int,
    ) -> list[int]:
        """Find x-positions where the full-height colour changes sharply."""
        band = 4
        col_means = arr.mean(axis=0).astype(float)

        splits: list[int] = []
        prev_band = col_means[:band].mean(axis=0)

        for x in range(band, w - band):
            curr_band = col_means[x : x + band].mean(axis=0)
            diff = float(np.linalg.norm(curr_band - prev_band))

            if diff > 10:
                left_strip = arr[:, max(0, x - 2) : x, :].mean(axis=1).astype(float)
                right_strip = arr[:, x : min(w, x + 2), :].mean(axis=1).astype(float)
                per_row_diff = np.linalg.norm(right_strip - left_strip, axis=1)
                consistent = float((per_row_diff > 8).mean())

                if consistent > 0.55 and (not splits or (x - splits[-1]) > 20):
                    splits.append(x)
                    prev_band = (
                        col_means[x + band : x + 2 * band].mean(axis=0)
                        if x + 2 * band < w
                        else curr_band
                    )
                    continue
            prev_band = curr_band

        return splits

    @staticmethod
    def _classify_structural_region(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        w: int,
        h: int,
    ) -> str:
        """Heuristic label based on position and aspect ratio."""
        rw = x2 - x1
        rh = y2 - y1
        aspect = rw / max(rh, 1)

        if y1 < h * 0.1 and rw > w * 0.8 and aspect > 4:
            return "header"
        if y2 > h * 0.9 and rw > w * 0.8 and aspect > 4:
            return "status_bar"
        if x1 < w * 0.05 and rh > h * 0.5 and rw < w * 0.35:
            return "sidebar"
        if x2 > w * 0.95 and rh > h * 0.5 and rw < w * 0.35:
            return "sidebar"
        return "content_area"

    # ══════════════════════════════════════════════════════════════════════
    #  Strategy 2: contour-based element detection (from old codebase)
    # ══════════════════════════════════════════════════════════════════════

    def _detect_rectangles(
        self,
        gray: np.ndarray,
        h: int,
        w: int,
    ) -> list[DetectionResult]:
        """Detect rectangular UI elements via adaptive thresholding + contour analysis."""
        detections: list[DetectionResult] = []

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (h * w) * 0.0005  # 0.05 % of image
        max_area = (h * w) * 0.8  # 80 % of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect_ratio = bw / max(bh, 1)
                label = self._classify_rectangle(bw, bh, aspect_ratio, w, h)

                detections.append(
                    DetectionResult(
                        label=label,
                        bbox=(x, y, x + bw, y + bh),
                        confidence=0.6,
                        meta={"source": "contour"},
                    )
                )

        return detections

    @staticmethod
    def _classify_rectangle(
        bw: int,
        bh: int,
        aspect_ratio: float,
        img_w: int,
        img_h: int,
    ) -> str:
        """Classify a rectangular contour based on size and aspect ratio."""
        rel_w = bw / img_w
        rel_h = bh / img_h

        # Button: small-medium, wider than tall
        if 0.03 < rel_w < 0.25 and 0.02 < rel_h < 0.08 and 1.5 < aspect_ratio < 8:
            return "button"
        # Text input: wider than tall, medium size
        if 0.1 < rel_w < 0.6 and 0.02 < rel_h < 0.06 and aspect_ratio > 4:
            return "text_input"
        # Large horizontal bar (navbar / menu)
        if rel_w > 0.7 and rel_h < 0.1:
            return "menu"
        # Modal: centered, medium-large
        if 0.2 < rel_w < 0.7 and 0.2 < rel_h < 0.7:
            return "modal"
        # Large area
        if rel_w > 0.3 or rel_h > 0.3:
            return "container"
        # Small square: icon / checkbox
        if 0.8 < aspect_ratio < 1.2 and rel_w < 0.05:
            return "icon"

        return "container"

    # ══════════════════════════════════════════════════════════════════════
    #  Strategy 3: text-region clustering (from old codebase)
    # ══════════════════════════════════════════════════════════════════════

    def _detect_text_regions(
        self,
        gray: np.ndarray,
        h: int,
        w: int,
    ) -> list[DetectionResult]:
        """Detect text-heavy regions using connected-components analysis."""
        detections: list[DetectionResult] = []

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        dilated = cv2.dilate(binary, kernel_h, iterations=1)
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            dilated,
            connectivity=8,
        )

        bboxes: list[list[int]] = []
        min_cc_area = 100
        max_cc_area = int(h * w * 0.3)

        for i in range(1, num_labels):  # skip background
            x, y, bw, bh, area = stats[i]
            if area < min_cc_area or area > max_cc_area:
                continue
            aspect_ratio = bw / max(bh, 1)
            if aspect_ratio < 0.2 or aspect_ratio > 50:
                continue
            bboxes.append([x, y, x + bw, y + bh])

        if not bboxes:
            return detections

        # Cluster nearby boxes into larger text blocks
        text_blocks = self._cluster_text_regions(bboxes, h, w)

        for block in text_blocks:
            x1, y1, x2, y2 = block
            detections.append(
                DetectionResult(
                    label="text_block",
                    bbox=(x1, y1, x2, y2),
                    confidence=0.5,
                    meta={"source": "text_cluster"},
                )
            )

        return detections

    @staticmethod
    def _cluster_text_regions(
        bboxes: list[list[int]],
        h: int,
        w: int,
    ) -> list[list[int]]:
        """Cluster nearby text bounding boxes into text-block regions."""
        bboxes_arr = np.array(bboxes)

        merge_dist_x = w * 0.05  # 5 % of image width
        merge_dist_y = h * 0.02  # 2 % of image height

        clusters: list[list[int]] = []
        used: set[int] = set()

        for i in range(len(bboxes_arr)):
            if i in used:
                continue

            cluster = [bboxes_arr[i]]
            used.add(i)

            for j in range(i + 1, len(bboxes_arr)):
                if j in used:
                    continue

                bi = bboxes_arr[i]
                bj = bboxes_arr[j]

                dx = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
                dy = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))

                if dx < merge_dist_x and dy < merge_dist_y:
                    cluster.append(bj)
                    used.add(j)

            cluster_arr = np.array(cluster)
            merged = [
                int(cluster_arr[:, 0].min()),
                int(cluster_arr[:, 1].min()),
                int(cluster_arr[:, 2].max()),
                int(cluster_arr[:, 3].max()),
            ]

            area = (merged[2] - merged[0]) * (merged[3] - merged[1])
            if area > (h * w) * 0.001:  # at least 0.1 % of image
                clusters.append(merged)

        return clusters

    # ══════════════════════════════════════════════════════════════════════
    #  Non-Maximum Suppression
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _nms(
        detections: list[DetectionResult],
        threshold: float,
    ) -> list[DetectionResult]:
        """Remove overlapping detections, keeping highest-confidence ones."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep: list[DetectionResult] = []
        for det in detections:
            should_keep = True
            for kept in keep:
                iou = _compute_iou(det.bbox, kept.bbox)
                if iou > threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)

        return keep


# ─── helpers ──────────────────────────────────────────────────────────────


def _compute_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
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
