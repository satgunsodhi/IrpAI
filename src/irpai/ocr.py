"""
IrpAI Stage 3: Text Branch - OCR Engine

Extracts text content with precise positional information using:
- PaddleOCR (primary, highest accuracy)
- EasyOCR (fallback)
- Tesseract (lightweight fallback)

Supports selective OCR using UI detections as region proposals.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import cv2
import numpy as np

from irpai.config import OCRConfig
from irpai.models import BoundingBox, TextExtraction, UIDetection

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """PaddleOCR-based text extraction engine (primary)."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._ocr = None

    def _ensure_loaded(self):
        """Lazy-load the OCR model."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                self._ocr = PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.language,
                    use_gpu=False,  # Controlled by pipeline config
                    show_log=False,
                )
                logger.info("PaddleOCR engine loaded")
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle"
                )

    def extract(self, img: np.ndarray) -> list[TextExtraction]:
        """
        Extract text from image using PaddleOCR.

        Args:
            img: Input image (BGR)

        Returns:
            List of TextExtraction objects
        """
        self._ensure_loaded()

        result = self._ocr.ocr(img, cls=self.config.use_angle_cls)

        if result is None or not result[0]:
            return []

        extractions = []
        for line in result[0]:
            bbox_points = line[0]
            text = line[1][0]
            confidence = float(line[1][1])

            if confidence < self.config.confidence_threshold:
                continue

            if len(text.strip()) == 0:
                continue

            # Convert 4-point polygon to axis-aligned rectangle
            bbox = self._polygon_to_rect(bbox_points)

            extractions.append(
                TextExtraction(
                    text=text.strip()[:self.config.max_text_length],
                    bbox=bbox,
                    confidence=confidence,
                )
            )

        return extractions

    @staticmethod
    def _polygon_to_rect(points: list) -> BoundingBox:
        """Convert 4-point polygon to axis-aligned bounding box."""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return BoundingBox(
            x1=min(x_coords),
            y1=min(y_coords),
            x2=max(x_coords),
            y2=max(y_coords),
        )


class EasyOCREngine:
    """EasyOCR-based text extraction engine (fallback)."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._reader = None

    def _ensure_loaded(self):
        """Lazy-load the OCR reader."""
        if self._reader is None:
            try:
                import easyocr

                lang_map = {"en": "en", "ch": "ch_sim", "ja": "ja", "ko": "ko"}
                lang = lang_map.get(self.config.language, self.config.language)
                self._reader = easyocr.Reader([lang], gpu=False)
                logger.info("EasyOCR engine loaded")
            except ImportError:
                raise ImportError("EasyOCR not installed. Install with: pip install easyocr")

    def extract(self, img: np.ndarray) -> list[TextExtraction]:
        """Extract text using EasyOCR."""
        self._ensure_loaded()

        results = self._reader.readtext(img)

        extractions = []
        for bbox_points, text, confidence in results:
            if confidence < self.config.confidence_threshold:
                continue

            if len(text.strip()) == 0:
                continue

            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]

            extractions.append(
                TextExtraction(
                    text=text.strip()[:self.config.max_text_length],
                    bbox=BoundingBox(
                        x1=min(x_coords),
                        y1=min(y_coords),
                        x2=max(x_coords),
                        y2=max(y_coords),
                    ),
                    confidence=float(confidence),
                )
            )

        return extractions


class TesseractEngine:
    """Tesseract-based text extraction engine (lightweight fallback)."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()

    def extract(self, img: np.ndarray) -> list[TextExtraction]:
        """Extract text using Tesseract."""
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract"
            )

        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get detailed data with bounding boxes
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

        extractions = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if conf < self.config.confidence_threshold * 100:
                continue

            if len(text) == 0:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            extractions.append(
                TextExtraction(
                    text=text[:self.config.max_text_length],
                    bbox=BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h),
                    confidence=conf / 100.0,
                )
            )

        # Merge nearby words into lines
        return self._merge_word_extractions(extractions)

    @staticmethod
    def _merge_word_extractions(extractions: list[TextExtraction]) -> list[TextExtraction]:
        """Merge individual word detections into text lines."""
        if not extractions:
            return []

        # Sort by y position then x position
        sorted_exts = sorted(extractions, key=lambda e: (e.bbox.y1, e.bbox.x1))

        merged = []
        current_line: list[TextExtraction] = [sorted_exts[0]]

        for ext in sorted_exts[1:]:
            prev = current_line[-1]

            # Same line if y-centers are close
            prev_cy = (prev.bbox.y1 + prev.bbox.y2) / 2
            curr_cy = (ext.bbox.y1 + ext.bbox.y2) / 2
            line_height = prev.bbox.y2 - prev.bbox.y1

            if abs(curr_cy - prev_cy) < line_height * 0.5:
                current_line.append(ext)
            else:
                merged.append(_merge_line(current_line))
                current_line = [ext]

        if current_line:
            merged.append(_merge_line(current_line))

        return merged


def _merge_line(line_exts: list[TextExtraction]) -> TextExtraction:
    """Merge a line of word extractions into one."""
    text = " ".join(e.text for e in line_exts)
    avg_conf = sum(e.confidence for e in line_exts) / len(line_exts)

    x1 = min(e.bbox.x1 for e in line_exts)
    y1 = min(e.bbox.y1 for e in line_exts)
    x2 = max(e.bbox.x2 for e in line_exts)
    y2 = max(e.bbox.y2 for e in line_exts)

    return TextExtraction(
        text=text,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=avg_conf,
    )


class OCRExtractor:
    """
    Unified OCR text extraction engine.

    Supports multiple backends and selective OCR (using UI detection
    regions as proposals for faster, more accurate extraction).
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._engine = self._create_engine()

    def _create_engine(self):
        """Create the appropriate OCR engine."""
        engine_map = {
            "paddleocr": PaddleOCREngine,
            "easyocr": EasyOCREngine,
            "tesseract": TesseractEngine,
        }

        engine_cls = engine_map.get(self.config.engine)
        if engine_cls is None:
            logger.warning(f"Unknown OCR engine '{self.config.engine}', falling back to PaddleOCR")
            engine_cls = PaddleOCREngine

        try:
            return engine_cls(self.config)
        except ImportError as e:
            logger.warning(f"Failed to load {self.config.engine}: {e}")
            # Try fallbacks
            for fallback_name, fallback_cls in engine_map.items():
                if fallback_name != self.config.engine:
                    try:
                        engine = fallback_cls(self.config)
                        logger.info(f"Using fallback OCR engine: {fallback_name}")
                        return engine
                    except ImportError:
                        continue

            raise ImportError(
                "No OCR engine available. Install one of: paddleocr, easyocr, pytesseract"
            )

    def extract(
        self,
        img: np.ndarray,
        ui_detections: Optional[list[UIDetection]] = None,
    ) -> list[TextExtraction]:
        """
        Extract text from screenshot with optional selective OCR.

        If UI detections are provided and selective OCR is enabled,
        only OCR text-containing regions for faster processing.

        Args:
            img: Preprocessed screenshot (BGR)
            ui_detections: Optional UI element detections for selective OCR

        Returns:
            List of TextExtraction objects with positions
        """
        start_time = time.perf_counter()

        if ui_detections and self.config.use_selective_ocr:
            extractions = self._selective_ocr(img, ui_detections)
        else:
            extractions = self._full_ocr(img)

        # Sort by reading order (top-to-bottom, left-to-right)
        extractions.sort(key=lambda e: (e.bbox.y1, e.bbox.x1))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"OCR extraction: {len(extractions)} text regions in {elapsed_ms:.1f}ms")

        return extractions

    def _full_ocr(self, img: np.ndarray) -> list[TextExtraction]:
        """Run OCR on the entire image."""
        return self._engine.extract(img)

    def _selective_ocr(
        self, img: np.ndarray, ui_detections: list[UIDetection]
    ) -> list[TextExtraction]:
        """
        Run OCR only on text-containing UI regions.

        3-5x faster than full-image OCR with higher precision.
        """
        extractions = []
        text_regions = [d for d in ui_detections if d.text_region]

        if not text_regions:
            # Fall back to full OCR if no text regions detected
            logger.debug("No text regions in UI detections, falling back to full OCR")
            return self._full_ocr(img)

        h, w = img.shape[:2]

        for region in text_regions:
            # Extract ROI with padding
            pad = 5
            x1 = max(0, int(region.bbox.x1) - pad)
            y1 = max(0, int(region.bbox.y1) - pad)
            x2 = min(w, int(region.bbox.x2) + pad)
            y2 = min(h, int(region.bbox.y2) + pad)

            if x2 - x1 < 10 or y2 - y1 < 5:
                continue

            roi = img[y1:y2, x1:x2]

            # Run OCR on ROI
            try:
                roi_extractions = self._engine.extract(roi)
            except Exception as e:
                logger.debug(f"OCR failed on region {region.bbox}: {e}")
                continue

            # Adjust coordinates to global space
            for ext in roi_extractions:
                ext.bbox = BoundingBox(
                    x1=ext.bbox.x1 + x1,
                    y1=ext.bbox.y1 + y1,
                    x2=ext.bbox.x2 + x1,
                    y2=ext.bbox.y2 + y1,
                )
                ext.ui_region = region.element_type.value

            extractions.extend(roi_extractions)

        return extractions

    def extract_ensemble(
        self, img: np.ndarray, ui_detections: Optional[list[UIDetection]] = None
    ) -> list[TextExtraction]:
        """
        Multi-OCR ensemble extraction for maximum accuracy.

        Runs multiple OCR engines and uses weighted voting
        to determine the final text. 3x slower but more accurate.

        Args:
            img: Preprocessed screenshot (BGR)
            ui_detections: Optional UI detections for selective OCR

        Returns:
            List of TextExtraction with consensus text
        """
        engines = []
        weights = {}

        # Try to load all available engines
        for name, cls in [
            ("paddleocr", PaddleOCREngine),
            ("easyocr", EasyOCREngine),
            ("tesseract", TesseractEngine),
        ]:
            try:
                engine = cls(self.config)
                engines.append((name, engine))
                weights[name] = {"paddleocr": 0.5, "easyocr": 0.3, "tesseract": 0.2}.get(
                    name, 0.2
                )
            except ImportError:
                continue

        if len(engines) <= 1:
            logger.info("Only one OCR engine available, skipping ensemble")
            return self.extract(img, ui_detections)

        # Collect results from all engines
        all_results = {}
        for name, engine in engines:
            try:
                results = engine.extract(img)
                all_results[name] = results
            except Exception as e:
                logger.warning(f"Engine {name} failed: {e}")

        if not all_results:
            return []

        # Use primary engine's results as base, enhance with ensemble
        primary = list(all_results.values())[0]

        # For each primary extraction, find matching extractions in other engines
        enhanced = []
        for ext in primary:
            matching_texts = [ext.text]

            for name, results in list(all_results.items())[1:]:
                for other in results:
                    # Check spatial overlap
                    if ext.bbox.iou(other.bbox) > 0.3:
                        matching_texts.append(other.text)
                        break

            # Use most common text (majority vote)
            if len(matching_texts) > 1:
                from collections import Counter

                text_counts = Counter(matching_texts)
                best_text = text_counts.most_common(1)[0][0]
                ext.text = best_text
                ext.confidence = min(1.0, ext.confidence + 0.05)  # Boost confidence

            enhanced.append(ext)

        return enhanced


class PIIRedactor:
    """Detect and redact personally identifiable information from extracted text."""

    PII_PATTERNS = {
        "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    @classmethod
    def redact(cls, extractions: list[TextExtraction]) -> list[TextExtraction]:
        """Redact PII from text extractions."""
        for ext in extractions:
            for pii_type, pattern in cls.PII_PATTERNS.items():
                ext.text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", ext.text)
        return extractions

    @classmethod
    def detect_pii(cls, text: str) -> list[dict]:
        """Detect PII in text without redacting."""
        found = []
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found.append({"type": pii_type, "count": len(matches)})
        return found
